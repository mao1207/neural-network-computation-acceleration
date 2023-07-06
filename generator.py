import torch
import torch.nn as nn
import numpy as np

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        # LCP
        self.conv11 = torch.nn.Conv2d(in_channels=2, out_channels=21, kernel_size=(1, 15), padding=(0, 7))
        self.bn11 = torch.nn.BatchNorm2d(21)
        self.relu11 = torch.nn.ReLU(inplace=True)
        self.conv12 = torch.nn.Conv2d(in_channels=2, out_channels=21, kernel_size=(1, 15), padding=(0, 7))
        self.bn12 = torch.nn.BatchNorm2d(21)
        self.relu12 = torch.nn.ReLU(inplace=True)
        self.conv13 = torch.nn.Conv2d(in_channels=2, out_channels=22, kernel_size=(1, 15), padding=(0, 7))
        self.bn13 = torch.nn.BatchNorm2d(22)
        self.relu13 = torch.nn.ReLU(inplace=True)

        self.conv21 = torch.nn.Conv2d(in_channels=21, out_channels=170, kernel_size=(1, 15), padding=(0, 7), stride=10)
        self.bn21 = torch.nn.BatchNorm2d(170)
        self.relu21 = torch.nn.ReLU(inplace=True)
        self.conv22 = torch.nn.Conv2d(in_channels=21, out_channels=170, kernel_size=(1, 15), padding=(0, 7), stride=10)
        self.bn22 = torch.nn.BatchNorm2d(170)
        self.relu22 = torch.nn.ReLU(inplace=True)
        self.conv23 = torch.nn.Conv2d(in_channels=22, out_channels=172, kernel_size=(1, 15), padding=(0, 7), stride=10)
        self.bn23 = torch.nn.BatchNorm2d(172)
        self.relu23 = torch.nn.ReLU(inplace=True)

        self.conv31 = torch.nn.Conv2d(in_channels=170, out_channels=170, kernel_size=(1, 15), padding=(0, 7))
        self.bn31 = torch.nn.BatchNorm2d(170)
        self.relu31 = torch.nn.ReLU(inplace=True)
        self.conv32 = torch.nn.Conv2d(in_channels=170, out_channels=170, kernel_size=(1, 15), padding=(0, 7))
        self.bn32 = torch.nn.BatchNorm2d(170)
        self.relu32 = torch.nn.ReLU(inplace=True)
        self.conv33 = torch.nn.Conv2d(in_channels=172, out_channels=172, kernel_size=(1, 15), padding=(0, 7))
        self.bn33 = torch.nn.BatchNorm2d(172)
        self.relu33 = torch.nn.ReLU(inplace=True)

        self.conv41 = torch.nn.Conv2d(in_channels=170, out_channels=170, kernel_size=(1, 15), padding=(0, 7))
        self.bn41 = torch.nn.BatchNorm2d(170)
        self.relu41 = torch.nn.ReLU(inplace=True)
        self.conv42 = torch.nn.Conv2d(in_channels=170, out_channels=170, kernel_size=(1, 15), padding=(0, 7))
        self.bn42 = torch.nn.BatchNorm2d(170)
        self.relu42 = torch.nn.ReLU(inplace=True)
        self.conv43 = torch.nn.Conv2d(in_channels=172, out_channels=172, kernel_size=(1, 15), padding=(0, 7))
        self.bn43 = torch.nn.BatchNorm2d(172)
        self.relu43 = torch.nn.ReLU(inplace=True)

        self.conv51 = torch.nn.Conv2d(in_channels=170, out_channels=682, kernel_size=(1, 15), padding=(0, 7), stride=10)
        self.bn51 = torch.nn.BatchNorm2d(682)
        self.relu51 = torch.nn.ReLU(inplace=True)
        self.conv52 = torch.nn.Conv2d(in_channels=170, out_channels=682, kernel_size=(1, 15), padding=(0, 7), stride=10)
        self.bn52 = torch.nn.BatchNorm2d(682)
        self.relu52 = torch.nn.ReLU(inplace=True)
        self.conv53 = torch.nn.Conv2d(in_channels=172, out_channels=684, kernel_size=(1, 15), padding=(0, 7), stride=10)
        self.bn53 = torch.nn.BatchNorm2d(684)
        self.relu53 = torch.nn.ReLU(inplace=True)

        self.conv61 = torch.nn.Conv2d(in_channels=682, out_channels=682, kernel_size=(1, 15), padding=(0, 7))
        self.bn61 = torch.nn.BatchNorm2d(682)
        self.relu61 = torch.nn.ReLU(inplace=True)
        self.conv62 = torch.nn.Conv2d(in_channels=682, out_channels=682, kernel_size=(1, 15), padding=(0, 7))
        self.bn62 = torch.nn.BatchNorm2d(682)
        self.relu62 = torch.nn.ReLU(inplace=True)
        self.conv63 = torch.nn.Conv2d(in_channels=684, out_channels=684, kernel_size=(1, 15), padding=(0, 7))
        self.bn63 = torch.nn.BatchNorm2d(684)
        self.relu63 = torch.nn.ReLU(inplace=True)

        self.conv71 = torch.nn.Conv2d(in_channels=682, out_channels=682, kernel_size=(1, 15), padding=(0, 7))
        self.conv72 = torch.nn.Conv2d(in_channels=682, out_channels=682, kernel_size=(1, 15), padding=(0, 7))
        self.conv73 = torch.nn.Conv2d(in_channels=684, out_channels=684, kernel_size=(1, 15), padding=(0, 7))

        self.bn7 = torch.nn.BatchNorm2d(2048)
        self.relu7 = torch.nn.ReLU(inplace=True)

        self.avgpool = torch.nn.AvgPool2d(kernel_size=(1, 8))
        self.flatten = torch.nn.Flatten()

        self.fc = torch.nn.Linear(2048, 4, bias=False)

    def forward(self, x):
        element = 0
        # LCP
        x1 = self.conv11(x[:, :2])
        x2 = self.conv12(x[:, 2:4])
        x3 = self.conv13(x[:, 4:])
        x1 = self.bn11(x1)
        x2 = self.bn12(x2)
        x3 = self.bn13(x3)
        x1 = self.relu11(x1)
        x2 = self.relu12(x2)
        x3 = self.relu13(x3)

        x1 = self.conv21(x1)
        x2 = self.conv22(x2)
        x3 = self.conv23(x3)
        x1 = self.bn21(x1)
        x2 = self.bn22(x2)
        x3 = self.bn23(x3)
        x1 = self.relu21(x1)
        x2 = self.relu22(x2)
        x3 = self.relu23(x3)
        tmp1 = x1
        tmp2 = x2
        tmp3 = x3

        x1 = self.conv31(x1)
        x2 = self.conv32(x2)
        x3 = self.conv33(x3)
        x1 = self.bn31(x1)
        x2 = self.bn32(x2)
        x3 = self.bn33(x3)
        x1 = self.relu31(x1)
        x2 = self.relu32(x2)
        x3 = self.relu33(x3)

        x1 = self.conv41(x1)
        x2 = self.conv42(x2)
        x3 = self.conv43(x3)
        x1 = self.bn41(x1) + tmp1
        x2 = self.bn42(x2) + tmp2
        x3 = self.bn43(x3) + tmp3
        x1 = self.relu41(x1)
        x2 = self.relu42(x2)
        x3 = self.relu43(x3)

        x1 = self.conv51(x1)
        x2 = self.conv52(x2)
        x3 = self.conv53(x3)
        x1 = self.bn51(x1)
        x2 = self.bn52(x2)
        x3 = self.bn53(x3)
        x1 = self.relu51(x1)
        x2 = self.relu52(x2)
        x3 = self.relu53(x3)

        x1 = self.conv61(x1)
        x2 = self.conv62(x2)
        x3 = self.conv63(x3)
        x1 = self.bn61(x1)
        x2 = self.bn62(x2)
        x3 = self.bn63(x3)
        x1 = self.relu61(x1)
        x2 = self.relu62(x2)
        x3 = self.relu63(x3)

        x1 = self.conv71(x1)
        x2 = self.conv72(x2)
        x3 = self.conv73(x3)

        element += (x1.numel() + x2.numel() + x3.numel())

        # 传输到中心节点
        x = torch.cat([x1, x2, x3], dim=1)

        x = self.bn7(x)
        x = self.relu7(x)

        x = self.avgpool(x)
        x = self.flatten(x)

        input = x

        x = self.fc(x)

        return input, x


X = torch.Tensor(np.load('./X_train.npy'))


net = torch.load("model.pth")
net.eval()
net_state_dict = net.state_dict()

resnet = Resnet()
resnet_state_dict = resnet.state_dict()
for k in net_state_dict.keys():
   if k in resnet_state_dict.keys():
       resnet_state_dict[k] = net_state_dict[k]
   else:
       print("no model")
resnet.load_state_dict(resnet_state_dict)
resnet.eval()

input, x = resnet(X)
torch.save(input, "train_matrices_signal.npy")

torch.save(net_state_dict["fc.weight"], "weight_signal.npy")
torch.save(net_state_dict["fc.bias"], "bias_signal.npy")