import torch
import torch.nn as nn

class YOLOPAFPN(nn.Module):
    def __init__(self):
        super(YOLOPAFPN, self).__init__()
        # ...existing code...

    def forward(self, x):
        # ...existing code...

        # Add debugging prints to check the shapes of the tensors before concatenation
        print(f"Shape of f_out0 before concatenation: {f_out0.shape}")
        print(f"Shape of x1 before concatenation: {x1.shape}")

        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16

        # ...existing code...

        return [f_out0, f_out1, f_out2]

# ...existing code...
