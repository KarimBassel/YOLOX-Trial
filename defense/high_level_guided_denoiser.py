"""
    Implemented HGD as demonstrated in the following paper
    https://openaccess.thecvf.com/content_cvpr_2018/html/Liao_Defense_Against_Adversarial_CVPR_2018_paper.html
    with modifications to the architecture as demonstrated in the following paper
    https://doi.org/10.48550/arXiv.1608.06993
"""
from torch.nn import Module, BatchNorm2d, ReLU, Conv2d, Sequential, ConvTranspose2d, Dropout2d
from torch.nn import ModuleDict
import torch.nn.functional as F
from torch import Tensor
import torch

class DenseLayer(Module):
    def __init__(self, num_input_features, growth_rate, bn_size) -> None:
        super(DenseLayer, self).__init__()

        # bottle neck
        self.layers = Sequential(
            BatchNorm2d(num_input_features),
            ReLU(inplace=True),
            Conv2d(
                in_channels=num_input_features,
                out_channels=bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            BatchNorm2d(bn_size * growth_rate),
            ReLU(inplace=True),
            Conv2d(
                in_channels=growth_rate * bn_size,
                out_channels=growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False)
        )

    def forward(self, x):
        if isinstance(x, Tensor):
            prev_features = torch.concat([x], 1)
        else:
            prev_features = torch.concat(x, 1)
        prev_features = torch.clamp(prev_features, min=-1e6, max=1e6)  # Clamp to avoid extreme values
        # print("DenseLayer - Input shape:", prev_features.shape, "Contains NaN:", torch.isnan(prev_features).any())
        out = self.layers(prev_features)
        out = torch.clamp(out, min=-1e6, max=1e6)  # Clamp output
        # print("DenseLayer - Output shape:", out.shape, "Contains NaN:", torch.isnan(out).any())
        return out


class DenseBlock(ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate) -> None:
        super(DenseBlock, self).__init__()

        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
            

    def forward(self, init_features):
        features = [init_features]
        # print("DenseBlock - Initial features shape:", init_features.shape, "Contains NaN:", torch.isnan(init_features).any())
        for name, layer in self.items():
            new_features = layer(features)
            # print(f"DenseBlock - Layer {name} output shape:", new_features.shape, "Contains NaN:", torch.isnan(new_features).any())
            features.append(new_features)
        concatenated_features = torch.concat(features, 1)
        # print("DenseBlock - Concatenated features shape:", concatenated_features.shape, "Contains NaN:", torch.isnan(concatenated_features).any())
        return concatenated_features


class Transition(Sequential):
    def __init__(self, num_input_features, num_output_features, kernel_size=2, stride=2, padding=0):
        super(Transition, self).__init__()
        self.add_module('norm', BatchNorm2d(num_input_features))
        self.add_module('relu', ReLU(inplace=True))
        self.add_module('conv', Conv2d(num_input_features, num_output_features,
                                       kernel_size=kernel_size, stride=stride, padding=1, bias=False))
        self.add_module('dropout', Dropout2d(p=0.1, inplace=True))

    def forward(self, x):
        x = torch.clamp(x, min=-1e6, max=1e6)  # Clamp input to Transition
        # print("Transition - Input shape:", x.shape, "Contains NaN:", torch.isnan(x).any())
        
        # Debug BatchNorm2d
        norm_out = self.norm(x)
        norm_out = torch.clamp(norm_out, min=-1e6, max=1e6)  # Clamp after BatchNorm2d
        # print("Transition - After BatchNorm2d shape:", norm_out.shape, "Contains NaN:", torch.isnan(norm_out).any())
        
        # Debug ReLU
        relu_out = self.relu(norm_out)
        relu_out = torch.clamp(relu_out, min=-1e6, max=1e6)  # Clamp after ReLU
        # print("Transition - After ReLU shape:", relu_out.shape, "Contains NaN:", torch.isnan(relu_out).any())
        
        # Add small epsilon to stabilize input
        relu_out = relu_out + 1e-8
        
        # Debug Conv2d weights
        # print("Transition - Conv2d weights contain NaN:", torch.isnan(self.conv.weight).any())
        # print("Transition - Conv2d weights max value:", self.conv.weight.max())
        # print("Transition - Conv2d weights min value:", self.conv.weight.min())
        
        # Debug Conv2d input
        # print("Transition - Conv2d input contains NaN:", torch.isnan(relu_out).any())
        
        # Conv2d operation
        conv_out = self.conv(relu_out)
        conv_out = torch.clamp(conv_out, min=-1e6, max=1e6)  # Clamp after Conv2d
        # print("Transition - After Conv2d shape:", conv_out.shape, "Contains NaN:", torch.isnan(conv_out).any())
        
        # Debug Dropout
        dropout_out = self.dropout(conv_out)
        dropout_out = torch.clamp(dropout_out, min=-1e6, max=1e6)  # Clamp after Dropout
        # print("Transition - After Dropout shape:", dropout_out.shape, "Contains NaN:", torch.isnan(dropout_out).any())
        
        return dropout_out

class Fuse(Module):
    def __init__(self) -> None:
        super(Fuse, self).__init__()

    def forward(self, small_image, large_image):
        small_image = torch.clamp(small_image, min=-1e6, max=1e6)  # Clamp small image
        large_image = torch.clamp(large_image, min=-1e6, max=1e6)  # Clamp large image
        # print("Fuse - Small image shape:", small_image.shape, "Contains NaN:", torch.isnan(small_image).any())
        # print("Fuse - Large image shape:", large_image.shape, "Contains NaN:", torch.isnan(large_image).any())
        # Use adaptive pooling to match dimensions
        target_size = large_image.shape[2:]
        upscaled_image = F.interpolate(small_image, size=target_size, mode="bilinear", align_corners=True)
        upscaled_image = torch.clamp(upscaled_image, min=-1e6, max=1e6)  # Clamp after interpolation
        # print("Fuse - Upscaled image shape:", upscaled_image.shape, "Contains NaN:", torch.isnan(upscaled_image).any())
        result_image = torch.cat((upscaled_image, large_image), dim=1)
        result_image = torch.clamp(result_image, min=-1e6, max=1e6)  # Clamp concatenated result
        # print("Fuse - Result image shape:", result_image.shape, "Contains NaN:", torch.isnan(result_image).any())
        return result_image  

class HGD(Module):
    def __init__(
            self,
            width = 1.0,
            growth_rate=32,
            bn_size=4,
            ) -> None:
        
        super(HGD, self).__init__()
        start_channels = int(width * 64)
        self.stem = Sequential(
            BatchNorm2d(3),
            ReLU(inplace=True),
            Conv2d(in_channels=3, out_channels=start_channels,
                    kernel_size=7, stride=2, padding=3, bias=False),
            Dropout2d(p=0.1,inplace=True),
            BatchNorm2d(start_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=start_channels, out_channels=start_channels,
                    kernel_size=3, stride=2, padding=1, bias=False),
            Dropout2d(p=0.1,inplace=True)
        )

        self.reverse_stem = Sequential(
            BatchNorm2d(start_channels),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=start_channels, out_channels=start_channels,
                   kernel_size=4, stride=2, padding=1,bias=False),

            BatchNorm2d(start_channels),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=start_channels, out_channels=start_channels,
                   kernel_size=4, stride=2, padding=1, bias=False),
        )
        

        self.fuse = Fuse()
        self.conv = Conv2d(in_channels=start_channels, out_channels=3, kernel_size=3, padding=1, stride=1)

        forward_path_info={
            "num_layers": [2, 3, 3, 3, 3],
            "layers_inputs": list(map(lambda x: int(x * width), [64, 64, 128, 256, 256])),
            "layers_outputs": list(map(lambda x: int(x * width), [64, 128, 256, 256, 256])), 
        }
        backward_path_info={
            "num_layers": [3, 3, 3, 2],
            "layers_inputs": list(map(lambda x: int(x * width),  [512, 512, 384, 192])),
            "layers_outputs": list(map(lambda x: int(x * width), [256, 256, 128, 64]))
        }

        for i, (num_layers, inp, out) in enumerate(
            zip(forward_path_info["num_layers"],
                forward_path_info["layers_inputs"],
                forward_path_info["layers_outputs"])):
            dense_block = DenseBlock(num_layers, inp, bn_size, growth_rate)
            if i == 0:
                kernel_size = 3
                padding = 1
                stride = 1
            else:
                kernel_size = 2
                padding = 2
                stride = 2

            # print(f"Creating Transition layer {i}:")
            # print(f"  Input channels: {inp + num_layers * growth_rate}, Output channels: {out}")
            # print(f"  Kernel size: {kernel_size}, Stride: {stride}, Padding: {padding}")

            transition = Transition(
                inp + num_layers * growth_rate, out,
                padding=padding, kernel_size=kernel_size, stride=stride)
            self.add_module(f"forward_{i}", dense_block)
            self.add_module(f"forward_transition_{i}", transition)

        for i, (num_layers, inp, out) in enumerate(
            zip(backward_path_info["num_layers"],
                backward_path_info["layers_inputs"],
                backward_path_info["layers_outputs"])):

            if i == len(backward_path_info["num_layers"]) - 1:
                kernel_size = 3
                padding = 1
                stride = 1
            else:
                kernel_size = 2
                padding = 2
                stride = 2


            dense_block = DenseBlock(num_layers, inp, bn_size, growth_rate)
            transition = Transition(
                inp + num_layers * growth_rate, out,
                  padding=padding, kernel_size=kernel_size, stride=stride)
            
            self.add_module(f"backward_{i}", dense_block)
            self.add_module(f"backward_transition_{i}", transition)
            
    def forward(self, input):
        input = torch.clamp(input, min=-1e6, max=1e6)  # Clamp input to the model
        # Backward path
        stem_out = self.stem(input)
        stem_out = torch.clamp(stem_out, min=-1e6, max=1e6)  # Clamp after stem
        # print("Stem output shape:", stem_out.shape, "Contains NaN:", torch.isnan(stem_out).any())
        out_forward = []
        out_forward.append(self.forward_transition_0(self.forward_0(stem_out)))
        # print("Forward path - Layer 0 output shape:", out_forward[0].shape, "Contains NaN:", torch.isnan(out_forward[0]).any())
        out_forward.append(self.forward_transition_1(self.forward_1(out_forward[0])))
        # print("Forward path - Layer 1 output shape:", out_forward[1].shape, "Contains NaN:", torch.isnan(out_forward[1]).any())
        out_forward.append(self.forward_transition_2(self.forward_2(out_forward[1])))
        # print("Forward path - Layer 2 output shape:", out_forward[2].shape, "Contains NaN:", torch.isnan(out_forward[2]).any())
        out_forward.append(self.forward_transition_3(self.forward_3(out_forward[2])))
        # print("Forward path - Layer 3 output shape:", out_forward[3].shape, "Contains NaN:", torch.isnan(out_forward[3]).any())
        out_forward.append(self.forward_transition_4(self.forward_4(out_forward[3])))
        # print("Forward path - Layer 4 output shape:", out_forward[4].shape, "Contains NaN:", torch.isnan(out_forward[4]).any())

        out_backward = self.fuse(out_forward[4], out_forward[3])
        # print("Backward path - After fuse with Layer 4 and Layer 3 output shape:", out_backward.shape, "Contains NaN:", torch.isnan(out_backward).any())
        out_backward = self.backward_transition_0(self.backward_0(out_backward))
        # print("Backward path - After backward transition 0 output shape:", out_backward.shape, "Contains NaN:", torch.isnan(out_backward).any())

        out_backward = self.fuse(out_backward, out_forward[2])
        # print("Backward path - After fuse with Layer 2 output shape:", out_backward.shape, "Contains NaN:", torch.isnan(out_backward).any())
        out_backward = self.backward_transition_1(self.backward_1(out_backward))
        # print("Backward path - After backward transition 1 output shape:", out_backward.shape, "Contains NaN:", torch.isnan(out_backward).any())

        out_backward = self.fuse(out_backward, out_forward[1])
        # print("Backward path - After fuse with Layer 1 output shape:", out_backward.shape, "Contains NaN:", torch.isnan(out_backward).any())
        out_backward = self.backward_transition_2(self.backward_2(out_backward))
        # print("Backward path - After backward transition 2 output shape:", out_backward.shape, "Contains NaN:", torch.isnan(out_backward).any())

        out_backward = self.fuse(out_backward, out_forward[0])
        # print("Backward path - After fuse with Layer 0 output shape:", out_backward.shape, "Contains NaN:", torch.isnan(out_backward).any())
        out_backward = self.backward_transition_3(self.backward_3(out_backward))
        # print("Backward path - After backward transition 3 output shape:", out_backward.shape, "Contains NaN:", torch.isnan(out_backward).any())

        out_backward = self.reverse_stem(out_backward)
        out_backward = torch.clamp(out_backward, min=-1e6, max=1e6)  # Clamp after reverse stem
        # print("Reverse stem output shape:", out_backward.shape, "Contains NaN:", torch.isnan(out_backward).any())
        out_backward = self.conv(out_backward)
        out_backward = torch.clamp(out_backward, min=-1e6, max=1e6)  # Clamp final output
        # print("Final output shape:", out_backward.shape, "Contains NaN:", torch.isnan(out_backward).any())
        return out_backward

if __name__ == "__main__":
    model = HGD(width=0.5)
    input = torch.randn((1, 3, 500, 500))
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))

    print(model(input).shape)