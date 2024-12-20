# base stuff
import os

# torch
import torch
import torch.distributed as dist

# custom stuff
from utils import metric

def inference(pargs, device, net, inference_loader, logger): 

    logger.log_start(key="inference_start")

    # Set model to evaluation mode 
    net.eval()

    # Use this to collect predictions for all inputs 
    predictions_all = []

    # Disable gradients 
    with torch.no_grad():
        # Iterate over inference samples 
        # !!!! I have duplicated the validation set for testing (see `data/__init__.py`) 
        for inputs_inf, filename_inf in inference_loader: 
            # Send inputs to device 
            inputs_inf = inputs_inf.to(device)

            # Forward pass 
            outputs_inf = net.forward(inputs_inf)

            # Post-process predictions (e.g., apply softmax and argmax for classification)
            predictions_inf = torch.argmax(torch.softmax(outputs_inf, 1), 1)

            # Collect predictions (you can save them to disk or return them)
            predictions_all.append((filename_inf, predictions_inf.cpu().numpy()))

    logger.log_end(key="inference_stop")
    
    # Return or save predictions
    return predictions_all

    # !!!! careful because it uses batch size 1? 