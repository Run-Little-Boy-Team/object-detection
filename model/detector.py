import torch
import torch.nn as nn

from .shufflenetv2 import ShuffleNetV2
from .custom_layers import DetectHead, SPP

class Detector(nn.Module):
    def __init__(self, category_num, load_param):
        super(Detector, self).__init__()

        self.category_num = category_num

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192]
        self.backbone = ShuffleNetV2(self.stage_repeats, self.stage_out_channels, load_param)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.SPP = SPP(sum(self.stage_out_channels[-3:]), self.stage_out_channels[-2])
         
        self.detect_head = DetectHead(self.stage_out_channels[-2], category_num)

    def forward(self, x):
        P1, P2, P3 = self.backbone(x)
        P3 = self.upsample(P3)
        P1 = self.avg_pool(P1)
        P = torch.cat((P1, P2, P3), dim=1)

        y = self.SPP(P)
        y = self.detect_head(y)

        return y
    
    def reshape_category_num(self, category_num):
        new_conv2d = torch.nn.Conv2d(self.stage_out_channels[-2], category_num, 1, stride=1, padding=0, bias=False)    
        new_bn2d = torch.nn.BatchNorm2d(category_num)
        if category_num >= self.category_num:
            print("Keeping old weights")
            new_conv2d.weight.data[:self.category_num, :, :, :] = self.detect_head.cls_layers.conv5x5[3].weight.data[:self.category_num, :, :, :]
            new_bn2d.weight.data[:self.category_num] = self.detect_head.cls_layers.conv5x5[4].weight.data[:self.category_num]
            new_bn2d.bias.data[:self.category_num] = self.detect_head.cls_layers.conv5x5[4].bias.data[:self.category_num]
            new_bn2d.running_mean.data[:self.category_num] = self.detect_head.cls_layers.conv5x5[4].running_mean.data[:self.category_num]
            new_bn2d.running_var.data[:self.category_num] = self.detect_head.cls_layers.conv5x5[4].running_var.data[:self.category_num]
        else:
            print("Droping old weights")  
        new_conv5x5 = torch.nn.Sequential(
            self.detect_head.cls_layers.conv5x5[0],
            self.detect_head.cls_layers.conv5x5[1],
            self.detect_head.cls_layers.conv5x5[2],
            new_conv2d,
            new_bn2d
        )
        device = next(self.parameters()).device
        self.detect_head.cls_layers.conv5x5 = new_conv5x5.to(device)
        self.category_num = category_num

    def fuse_modules(self, qat=False, inplace=False):
        self.eval()
        fusing_list = [
            line.strip().split(";")
            for line in open("fusing_list.csv", "r").readlines()
        ]
        if qat:
            fuse_function = torch.ao.quantization.fuse_modules_qat
        else:
            fuse_function = torch.ao.quantization.fuse_modules
        if inplace:
            fuse_function(
                self,
                fusing_list,
                inplace=inplace,
            )
        else:
            return fuse_function(
                self,
                fusing_list,
                inplace=inplace,
            )
        

if __name__ == "__main__":
    model = Detector(80, False)
    test_data = torch.rand(1, 3, 352, 352)
    torch.onnx.export(model,                    #model being run
                     test_data,                 # model input (or a tuple for multiple inputs)
                     "./test.onnx",             # where to save the model (can be a file or file-like object)
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=11,          # the ONNX version to export the model to
                     do_constant_folding=True)  # whether to execute constant folding for optimization

