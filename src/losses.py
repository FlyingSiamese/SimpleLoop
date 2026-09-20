import torch
import torch.nn.functional as F

def loop_window_loss(predictions,target,loss_window):
    losses = [F.mse_loss(predictions[t],target)
              for t in range(predictions.shape[0]-loss_window,predictions.shape[0])
            ]
    return torch.stack(losses).mean()