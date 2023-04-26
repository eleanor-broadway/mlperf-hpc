# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gc
import os
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch

from openfold.checkpoint_utils import (
    resume_from_latest_checkpoint,
    save_checkpoint_from_training,
)
from openfold.config import AlphaFoldConfig
from openfold.dataloaders import InitialTrainingDataloader, ValidationDataloader
from openfold.datasets import InitialTrainingDataset, ValidationDataset
from openfold.distributed import dist_gather_val_metrics, dist_reduce_losses_avg
from openfold.helpers import get_seed_from_string, get_timestamp_string, map_dict_values
from openfold.log_utils import save_logs
from openfold.loss import AlphaFoldLoss
from openfold.lr_scheduler import AlphaFoldLRScheduler
from openfold.model.alphafold import AlphaFold
from openfold.numpy_utils import NUMPY_SEED_MODULUS
from openfold.samplers import InitialTrainingSampler, ValidationSampler
from openfold.swa import AlphaFoldSWA
from openfold.torch_utils import (
    TORCH_SEED_MODULUS,
    disable_tf32,
    enable_tf32,
    map_tensor_tree,
)
from openfold.validation_metrics import compute_validation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_dirpath",
        type=Path,
        required=True,
        help="Path to training output directory.",
    )
    parser.add_argument(
        "--pdb_mmcif_chains_filepath",
        type=Path,
        required=True,
        help="Path to mmcif chains CSV file generated by data preprocessing.",
    )
    parser.add_argument(
        "--pdb_mmcif_dicts_dirpath",
        type=Path,
        required=True,
        help="Path to mmcif dicts directory generated by data preprocessing.",
    )
    parser.add_argument(
        "--pdb_obsolete_filepath",
        type=Path,
        required=True,
        help="Path to `obsolete.dat` file.",
    )
    parser.add_argument(
        "--pdb_alignments_dirpath",
        type=Path,
        required=True,
        help="Path to PDB alignments directory generated by data preprocessing.",
    )
    parser.add_argument(
        "--train_max_pdb_release_date",
        type=str,
        default="2021-09-16",
        help="Max PDB release date for training.",
    )
    parser.add_argument(
        "--val_min_cameo_submission_date",
        type=str,
        default="2021-09-17",
        help="Min submission date for CAMEO validation.",
    )
    parser.add_argument(
        "--val_max_cameo_submission_date",
        type=str,
        default="2021-12-11",
        help="Max submission date for CAMEO validation.",
    )
    parser.add_argument(
        "--val_max_sequence_length",
        type=int,
        default=700,
        help="Max sequence length for filtering CAMEO validation set.",
    )
    parser.add_argument(
        "--target_avg_lddt_ca_value",
        type=float,
        default=0.8,
        help="Target avg lDDT-Ca value required to stop training.",
    )
    parser.add_argument(
        "--initialize_parameters_from",
        type=Path,
        default=None,
        help="""Optional path to `.pt` checkpoint file
        used for parameter initialization.""",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "tf32", "bf16", "fp16", "amp"],
        default="tf32",
        help="Numerical precision.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="1234567890",
        help="Global seed for pseudorandom number generators.",
    )
    parser.add_argument(
        "--num_train_iters",
        type=int,
        default=80000,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--log_every_iters",
        type=int,
        default=-1,
        help="""Save logs every given iteration.
        A non-positive value disables the log saving.""",
    )
    parser.add_argument(
        "--checkpoint_every_iters",
        type=int,
        default=0,
        help="""Save checkpoints every given iteration.
        A non-positive value disables the checkpoint saving.""",
    )
    parser.add_argument(
        "--keep_last_checkpoints",
        type=int,
        default=0,
        help="How many last checkpoints to keep.",
    )
    parser.add_argument(
        "--val_every_iters",
        type=int,
        default=200,
        help="Compute validation every given iteration.",
    )
    parser.add_argument(
        "--keep_best_checkpoints",
        type=int,
        default=0,
        help="How many best checkpoints to keep.",
    )
    parser.add_argument(
        "--keep_val_checkpoints",
        action="store_true",
        help="Whether to keep all validation checkpoints.",
    )
    parser.add_argument(
        "--device_batch_size",
        type=int,
        default=1,
        help="Local batch size (per device).",
    )
    parser.add_argument(
        "--init_lr",
        type=float,
        default=1e-3,
        help="Initial learning rate value.",
    )
    parser.add_argument(
        "--final_lr",
        type=float,
        default=5e-5,
        help="Final learning rate value.",
    )
    parser.add_argument(
        "--warmup_lr_length",
        type=int,
        default=1000,
        help="Num iterations for learning rate warm-up.",
    )
    parser.add_argument(
        "--init_lr_length",
        type=int,
        default=60000,
        help="""Num iterations after which decrease
        the initial learning rate to its final value.""",
    )
    parser.add_argument(
        "--gradient_clipping",
        action="store_true",
        help="Whether to enable gradient clipping.",
    )
    parser.add_argument(
        "--clip_grad_max_norm",
        type=float,
        default=0.1,
        help="Max norm value for gradient clipping.",
    )
    parser.add_argument(
        "--gradient_accumulation_iters",
        type=int,
        default=1,
        help="""Gradient accumulation iters.
        The default value of 1 means no accumulation.
        When set to > 1, other _iters and _length args must be scaled accordingly.""",
    )
    parser.add_argument(
        "--num_train_dataloader_workers",
        type=int,
        default=14,
        help="Num workers (subprocesses) for each instance of training dataloader.",
    )
    parser.add_argument(
        "--num_val_dataloader_workers",
        type=int,
        default=2,
        help="Num workers (subprocesses) for each instance of validation dataloader.",
    )
    parser.add_argument(
        "--filter_by_alignments",
        action="store_true",
        help="Whether to filter out mmcif chains with no alignments.",
    )
    parser.add_argument(
        "--use_only_pdb_chain_ids",
        type=str,
        nargs="*",
        default=None,
        help="""Optional list of pdb chain ids
        for intersection with train and val datasets.""",
    )
    parser.add_argument(
        "--save_process_logs",
        action="store_true",
        help="Whether to save logs from each process.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Whether to enable distributed training.",
    )
    args = parser.parse_args()
    # saving checkpoints must coincide with validation:
    if args.checkpoint_every_iters > 0:
        assert args.val_every_iters % args.checkpoint_every_iters == 0
    # saving logs must coincide with validation and checkpoints:
    if args.log_every_iters > 0:
        assert args.val_every_iters % args.log_every_iters == 0
        if args.checkpoint_every_iters > 0:
            assert args.checkpoint_every_iters % args.log_every_iters == 0
    # everything must be divisble by gradient accumulation length:
    assert args.gradient_accumulation_iters >= 1
    assert args.num_train_iters % args.gradient_accumulation_iters == 0
    assert args.val_every_iters % args.gradient_accumulation_iters == 0
    assert args.checkpoint_every_iters % args.gradient_accumulation_iters == 0
    assert args.log_every_iters % args.gradient_accumulation_iters == 0
    assert args.warmup_lr_length % args.gradient_accumulation_iters == 0
    assert args.init_lr_length % args.gradient_accumulation_iters == 0
    return args


def create_alphafold_module(
    alphafold_config: AlphaFoldConfig,
    device: torch.device,
    seed: int,
) -> AlphaFold:
    numpy_random_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    torch_cuda_rng_state = torch.cuda.get_rng_state(device=device)
    np.random.seed(seed % NUMPY_SEED_MODULUS)
    torch.manual_seed(seed)
    alphafold = AlphaFold(config=alphafold_config)
    alphafold.to(device=device)
    torch.cuda.set_rng_state(torch_cuda_rng_state, device=device)
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(numpy_random_state)
    return alphafold


def initialize_parameters_from_checkpoint(
    alphafold: AlphaFold,
    optimizer: torch.optim.Optimizer,
    checkpoint_filepath: Path,
    device: torch.device,
    verbose: bool,
) -> None:
    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    is_resumable_checkpoint = bool(
        "alphafold_state_dict" in checkpoint and "optimizer_state_dict" in checkpoint
    )
    if is_resumable_checkpoint:
        init_alphafold_state_dict = checkpoint["alphafold_state_dict"]
        init_optimizer_state_dict = checkpoint["optimizer_state_dict"]
    else:
        init_alphafold_state_dict = checkpoint
        init_optimizer_state_dict = None
    if verbose:
        print(f"Initializing parameters from {repr(checkpoint_filepath)}...")
    alphafold.load_state_dict(init_alphafold_state_dict, strict=True)
    if verbose:
        print(f"Parameters initialized from {repr(checkpoint_filepath)} successfully!")
    if init_optimizer_state_dict is not None:
        if verbose:
            print(f"Initializing optimizer from {repr(checkpoint_filepath)}...")
        optimizer.load_state_dict(init_optimizer_state_dict)
        if verbose:
            print(
                f"Optimizer initialized from {repr(checkpoint_filepath)} successfully!"
            )


def validation(
    alphafold: Union[AlphaFold, AlphaFoldSWA],
    validation_dataloader: ValidationDataloader,
    device: torch.device,
) -> List[dict]:
    alphafold.eval()
    val_metrics_list = []
    val_batch_iterator = iter(validation_dataloader)
    for _ in range(len(validation_dataloader)):
        perf = -time.perf_counter()
        val_batch = next(val_batch_iterator)
        assert len(val_batch["id"]) == 1
        id_tuple = val_batch["id"][0]
        with torch.no_grad():
            val_batch = map_tensor_tree(
                fn=lambda t: t.to(device=device),
                tree=val_batch,
            )
            val_outputs = alphafold(val_batch)
            val_batch = map_tensor_tree(fn=lambda t: t[..., -1], tree=val_batch)
            val_metrics = compute_validation_metrics(
                predicted_atom_positions=val_outputs["final_atom_positions"],
                target_atom_positions=val_batch["all_atom_positions"],
                atom_mask=val_batch["all_atom_mask"],
                metrics_names={"lddt_ca"},
            )
        perf += time.perf_counter()
        val_metrics = map_dict_values(fn=lambda t: t.item(), d=val_metrics)
        val_metrics["val_index"] = id_tuple[1]
        val_metrics["pdb_chain_id"] = id_tuple[3]
        val_metrics["duration"] = perf
        val_metrics_list.append(val_metrics)
    alphafold.train()
    return val_metrics_list


def training(args: argparse.Namespace) -> None:
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if torch.distributed.is_initialized():
        # Assuming distributed training:
        assert args.distributed is True
        # https://pytorch.org/docs/stable/elastic/run.html#environment-variables
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        main_rank = 0
        is_main_process = bool(rank == main_rank)
        process_name = f"dist_process_rank{rank}"
        device = torch.device(f"cuda:{local_rank}")
        global_batch_size = args.device_batch_size * world_size
        if is_main_process:
            print(f"initialized distributed training: WORLD_SIZE={world_size}")
    else:
        # Assuming single GPU training:
        print("single GPU training")
        assert args.distributed is False
        rank = None
        world_size = None
        local_rank = None
        main_rank = None
        is_main_process = True
        process_name = "single_process"
        device = torch.device("cuda:0")
        global_batch_size = args.device_batch_size

    # Set device:
    torch.cuda.set_device(device=device)

    # Numerical precision settings:
    if args.precision == "fp32":
        disable_tf32()
    elif args.precision == "tf32":
        enable_tf32()
    elif args.precision in {"bf16", "fp16", "amp"}:
        raise NotImplementedError(f"precision={repr(args.precision)}")
    else:
        raise ValueError(f"unknown precision={repr(args.precision)}")

    # Get alphafold config:
    alphafold_config = AlphaFoldConfig.from_preset(
        stage="initial_training",
        precision=args.precision,
    )

    # Create training dataset:
    initial_training_dataset = InitialTrainingDataset(
        pdb_mmcif_chains_filepath=args.pdb_mmcif_chains_filepath,
        pdb_mmcif_dicts_dirpath=args.pdb_mmcif_dicts_dirpath,
        pdb_obsolete_filepath=args.pdb_obsolete_filepath,
        pdb_alignments_dirpath=args.pdb_alignments_dirpath,
        max_pdb_release_date=args.train_max_pdb_release_date,
        alphafold_config=alphafold_config,
        filter_by_alignments=args.filter_by_alignments,
        use_only_pdb_chain_ids=args.use_only_pdb_chain_ids,
        name=f"initial_training_dataset_{process_name}",
    )

    # Create validation dataset:
    validation_dataset = ValidationDataset(
        pdb_mmcif_chains_filepath=args.pdb_mmcif_chains_filepath,
        pdb_mmcif_dicts_dirpath=args.pdb_mmcif_dicts_dirpath,
        pdb_obsolete_filepath=args.pdb_obsolete_filepath,
        pdb_alignments_dirpath=args.pdb_alignments_dirpath,
        min_cameo_submission_date=args.val_min_cameo_submission_date,
        max_cameo_submission_date=args.val_max_cameo_submission_date,
        max_sequence_length=args.val_max_sequence_length,
        alphafold_config=alphafold_config,
        filter_by_alignments=args.filter_by_alignments,
        use_only_pdb_chain_ids=args.use_only_pdb_chain_ids,
        name=f"validation_dataset_{process_name}",
    )

    # Create alphafold module:
    alphafold = create_alphafold_module(
        alphafold_config=alphafold_config,
        device=device,
        seed=get_seed_from_string(f"alphafold_init_{args.seed}"),
    )
    alphafold.train()

    # Create alphafold loss module:
    alphafold_loss = AlphaFoldLoss(config=alphafold_config.loss_config)

    # Create optimizer:
    optimizer = torch.optim.Adam(
        params=alphafold.parameters(),
        lr=args.init_lr,  # lr is controlled by AlphaFoldLRScheduler
        eps=1e-6,
    )

    # Initialize parameters from checkpoint if provided:
    if args.initialize_parameters_from is not None:
        initialize_parameters_from_checkpoint(
            alphafold=alphafold,
            optimizer=optimizer,
            checkpoint_filepath=args.initialize_parameters_from,
            device=device,
            verbose=is_main_process,
        )

    # Create optional SWA version of AlphaFold for evaluation and checkpoints:
    swa_alphafold = AlphaFoldSWA(
        alphafold=alphafold,
        enabled=alphafold_config.swa_enabled,
        decay_rate=alphafold_config.swa_decay_rate,
    )

    # Resume from latest checkpoint if it exists:
    num_prev_iters = resume_from_latest_checkpoint(
        alphafold=alphafold,
        optimizer=optimizer,
        swa_alphafold=swa_alphafold,
        training_dirpath=args.training_dirpath,
        device=device,
        verbose=is_main_process,
    )
    assert num_prev_iters % args.gradient_accumulation_iters == 0

    # Distributed wrapper:
    if args.distributed:
        alphafold = torch.nn.parallel.DistributedDataParallel(module=alphafold)

    # Create training sampler:
    initial_training_sampler = InitialTrainingSampler(
        dataset=initial_training_dataset,
        device_batch_size=args.device_batch_size,
        global_batch_size=global_batch_size,
        num_train_iters=args.num_train_iters,
        seed=get_seed_from_string(f"initial_training_sampler_{args.seed}"),
        is_distributed=args.distributed,
        rank=rank,
        world_size=world_size,
        num_prev_iters=num_prev_iters,
    )

    # Create validation sampler:
    validation_sampler = ValidationSampler(
        dataset=validation_dataset,
        is_distributed=args.distributed,
        rank=rank,
        world_size=world_size,
    )

    # Create training dataloader:
    initial_training_dataloader = InitialTrainingDataloader(
        dataset=initial_training_dataset,
        sampler=initial_training_sampler,
        device_batch_size=args.device_batch_size,
        num_workers=args.num_train_dataloader_workers,
        seed=get_seed_from_string(f"initial_training_dataloader_{args.seed}"),
        uniform_recycling_iters=list(
            range(0, alphafold_config.num_recycling_iters + 1)
        ),
        gradient_accumulation_iters=args.gradient_accumulation_iters,
        num_prev_iters=num_prev_iters,
    )
    train_batch_iterator = iter(initial_training_dataloader)

    # Create validation dataloader:
    validation_dataloader = ValidationDataloader(
        dataset=validation_dataset,
        sampler=validation_sampler,
        num_workers=args.num_val_dataloader_workers,
    )

    # Create logging-related objects:
    train_logs = []
    process_logs = []
    logs_dirpath = args.training_dirpath / "logs"
    train_logs_outpath = logs_dirpath / "training.log"
    process_logs_outpath = logs_dirpath / (process_name + ".log")
    val_logs_outpath = logs_dirpath / "validation.log"
    is_logging_enabled = bool(args.log_every_iters > 0)
    is_main_process_and_logging = bool(is_main_process and is_logging_enabled)

    # Set first iteration:
    first_iteration = num_prev_iters + 1

    # Create learning rate scheduler:
    lr_scheduler = AlphaFoldLRScheduler(
        init_lr=args.init_lr,
        final_lr=args.final_lr,
        warmup_lr_length=args.warmup_lr_length,
        init_lr_length=args.init_lr_length,
        optimizer=optimizer,
        iteration=first_iteration,
        verbose=False,
    )

    # Training seed:
    training_seed = get_seed_from_string(f"training_seed_{args.seed}")

    # Training loop:
    for iteration in range(first_iteration, args.num_train_iters + 1):
        # Start training iteration perf measurement:
        perf_training = -time.perf_counter()

        # Deterministic forward pass during training (dropout etc.):
        torch.manual_seed((training_seed + iteration) % TORCH_SEED_MODULUS)

        # Next train batch:
        train_batch = next(train_batch_iterator)
        train_batch = map_tensor_tree(
            fn=lambda t: t.to(device=device),
            tree=train_batch,
        )
        num_recycling_iters = train_batch["aatype"].shape[-1] - 1

        # Forward pass:
        train_outputs = alphafold(train_batch)
        loss, losses = alphafold_loss(
            outputs=train_outputs,
            batch=map_tensor_tree(fn=lambda t: t[..., -1], tree=train_batch),
        )
        loss = loss / args.gradient_accumulation_iters

        # Backward pass:
        if (iteration - 1) % args.gradient_accumulation_iters == 0:
            optimizer.zero_grad()
        loss.backward()

        if iteration % args.gradient_accumulation_iters == 0:
            # Gradient clipping:
            if args.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    parameters=alphafold.parameters(),
                    max_norm=args.clip_grad_max_norm,
                )

            # Optimizer step (weights/parameters update):
            optimizer.step()

            # SWA update:
            if swa_alphafold.enabled:
                swa_alphafold.update(alphafold)

        # Average losses from distributed training:
        if is_logging_enabled:
            if args.distributed:
                losses_avg = dist_reduce_losses_avg(
                    losses=losses,
                    is_main_process=is_main_process,
                    main_rank=main_rank,
                    device=device,
                    synchronize=False,
                )
            else:
                losses_avg = losses
            # Convert losses from Dict[str, torch.Tensor] to Dict[str, float]:
            losses = map_dict_values(fn=lambda t: t.item(), d=losses)
            if is_main_process:
                losses_avg = map_dict_values(fn=lambda t: t.item(), d=losses_avg)

        # Finalize training iteration perf measurement:
        perf_training += time.perf_counter()

        # Update process logs:
        if is_logging_enabled and args.save_process_logs:
            process_log = {
                "iteration": iteration,
                "sample_ids": list(map(list, train_batch["id"])),
                "num_recycling_iters": num_recycling_iters,
                "timestamp": get_timestamp_string(),
                **{f"losses.{k}": v for k, v in losses.items()},
                "duration": perf_training,
            }
            process_logs.append(process_log)

        # Update train logs:
        if is_main_process_and_logging:
            train_log = {
                "iteration": iteration,
                "global_batch_size": global_batch_size,
                "num_recycling_iters": num_recycling_iters,
                "timestamp": get_timestamp_string(),
                **{f"losses_avg.{k}": v for k, v in losses_avg.items()},
                "duration": perf_training,
            }
            train_logs.append(train_log)
            print(f"training {train_log}")

        # Save process and train logs:
        if is_logging_enabled and iteration % args.log_every_iters == 0:
            if args.save_process_logs:
                save_logs(process_logs, process_logs_outpath, append=True)
            process_logs.clear()
            if is_main_process:
                save_logs(train_logs, train_logs_outpath, append=True)
                train_logs.clear()

        # Validation (evaluation):
        is_validation = bool(iteration % args.val_every_iters == 0)
        if is_validation:
            perf_validation = -time.perf_counter()
            if is_main_process_and_logging:
                print("validation...")
            del train_batch, train_outputs, loss
            # Execute validation (evaluation) loop:
            val_metrics_list = validation(
                alphafold=swa_alphafold if swa_alphafold.enabled else alphafold,
                validation_dataloader=validation_dataloader,
                device=device,
            )
            if args.distributed:
                # Collect per-sample validation metrics to main process:
                val_metrics_list = dist_gather_val_metrics(
                    val_metrics_list=val_metrics_list,
                    val_pdb_chain_ids=validation_dataset.pdb_chain_ids,
                    is_main_process=is_main_process,
                    main_rank=main_rank,
                    world_size=world_size,
                    device=device,
                    synchronize=True,
                )
            perf_validation += time.perf_counter()
            if is_main_process:
                # Compute aggregated validation metrics in main process:
                val_metrics_df = pd.DataFrame(val_metrics_list)
                val_avg_lddt_ca = float(val_metrics_df["lddt_ca"].mean())
                val_size = len(val_metrics_list)
                assert val_size == len(validation_dataset)
                val_throughput = val_size / perf_validation
            if is_main_process_and_logging:
                # Save validation logs:
                val_log = {
                    "iteration": iteration,
                    "avg_lddt_ca": val_avg_lddt_ca,
                    "timestamp": get_timestamp_string(),
                    "duration": perf_validation,
                    "size": val_size,
                    "throughput": val_throughput,
                }
                print(f"validation {val_log}")
                val_log["metrics_list"] = val_metrics_list
                save_logs([val_log], val_logs_outpath, append=True)
            # Check if validation reaches target accuracy:
            if is_main_process:
                if val_avg_lddt_ca >= args.target_avg_lddt_ca_value:
                    stop_training_flag = torch.ones(1, device=device)
                else:
                    stop_training_flag = torch.zeros(1, device=device)
            else:
                stop_training_flag = torch.zeros(1, device=device)
            if args.distributed:
                torch.distributed.broadcast(tensor=stop_training_flag, src=main_rank)
            # Preventively clear the cache created during validation:
            gc.collect()
            torch.cuda.empty_cache()

        # Save checkpoint:
        if (
            is_main_process
            and args.checkpoint_every_iters > 0
            and iteration % args.checkpoint_every_iters == 0
        ):
            save_checkpoint_from_training(
                alphafold=alphafold,
                optimizer=optimizer,
                swa_alphafold=swa_alphafold,
                iteration=iteration,
                training_dirpath=args.training_dirpath,
                keep_last_checkpoints=args.keep_last_checkpoints,
                keep_best_checkpoints=args.keep_best_checkpoints,
                keep_val_checkpoints=args.keep_val_checkpoints,
                is_validation=is_validation,
                val_avg_lddt_ca=val_avg_lddt_ca if is_validation else None,
            )

        # Stop training if reached target validation metric:
        if is_validation and stop_training_flag:
            break

        # LR scheduler update:
        lr_scheduler.step()

    # Synchronize before return:
    if args.distributed:
        torch.distributed.barrier()


if __name__ == "__main__":
    try:
        training(parse_args())
    except KeyboardInterrupt:
        print("KeyboardInterrupt... exit(1)")
        exit(1)
