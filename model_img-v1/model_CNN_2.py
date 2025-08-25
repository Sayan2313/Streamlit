import torch
from torch import nn
from torch import load,argmax,softmax
from pathlib import Path
import json
# Model 2
def conv_block(in_channels:int,out_channels:int,kernel_size,stride,
               padding,bias):
    seq = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding = padding,
                            bias = bias
        ),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding = padding,
                            bias = bias
        ),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(3,3),stride=(1,1))
        )
    return seq
def classifier_block(input: int,output : int):
    return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input,output)
        )
class model_CNN_2(nn.Module):
    def __init__(self,model_name : str,output_classes:int,in_channels : int = 3):
        super().__init__()
        self.model_name = model_name
        self.layer1_1 = conv_block(in_channels,32,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=True)
        self.classifier = classifier_block(32 * 14 * 14,output_classes)
    def forward(self,x):
        out = self.layer1_1(x)
        out = self.classifier(out)
        return out
# ENV
_HERE = Path(__file__).resolve().parent
_CKPT = _HERE / "best_model.pt"               
_NAMES = _HERE / "names_of_the_animals.json" 
with _NAMES.open("r", encoding="utf-8") as f:
    _IDX2NAME = json.load(f)
# Functions
def getModelObject():
    """
        Needs 64x64 pixels images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_CNN_2('dummy',90,3)
    pre_model_dict = load(str(_CKPT),map_location='cpu')
    model.load_state_dict(pre_model_dict['model_state_dict'],strict=True)
    model.to(device).eval()
    return model,device
def predToClass(logits) -> str:
    softmax_logits = softmax(logits,dim=1)
    idx = argmax(softmax_logits,dim=1)
    return _IDX2NAME[str(idx.item())]

        