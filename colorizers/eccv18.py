import torch
import torch.nn as nn
import numpy as np
from IPython import embed
from colorizers.eccv16 import ECCVGenerator, eccv16
import matplotlib.pyplot as plt

from colorizers.base_color import *
from colorizers import *


import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import argparse

class Net(BaseColor):
    # laoding the model
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(Net, self).__init__()

        self.model = eccv16(pretrained=True)
    #freaz all the parameters in previous layers
        for param in self.model.parameters():
            param.requires_grad = False


        self.fc = nn.Conv2d(2, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)

        self.criterion = nn.CrossEntropyLoss()
        #updating the weights  during training to minimize the loss function and
        self.optimizer = optim.SGD(self.fc.parameters(), lr=0.001, momentum=0.9)

        # self.train_loader = DataLoader(training_data, batch_size=64, shuffle=True)

    def full_train(self, num_epochs=5):
        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(self.train_loader)}")

        print("Finished Training")

    def forward(self, input_l):
        conv1_2 = self.model.model1(self.normalize_l(input_l))
        conv2_2 = self.model.model2(conv1_2)
        conv3_3 = self.model.model3(conv2_2)
        conv4_3 = self.model.model4(conv3_3)
        conv5_3 = self.model.model5(conv4_3)
        conv6_3 = self.model.model6(conv5_3)
        conv7_3 = self.model.model7(conv6_3)
        conv8_3 = self.model.model8(conv7_3)
        out_reg = self.model.model_out(self.model.softmax(conv8_3))

        final_out = self.fc(out_reg)


        # final_out = self.model.model8(out_reg)

        return self.unnormalize_ab(self.model.upsample4(final_out))

def eccv18(pretrained=True):
	model = Net()
	if(pretrained):
		import torch.utils.model_zoo as model_zoo
		model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
	return model


colorizer_eccv16 = Net().eval()
print()
parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='../imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if(opt.use_gpu):
	colorizer_eccv16.cuda()


# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
# out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

plt.imsave('%s_eccv18.png'%opt.save_prefix, out_img_eccv16)
# plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2,2,3)
#plt.imshow(out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off')

# plt.subplot(2,2,4)
# plt.imshow(out_img_siggraph17)
# plt.title('Output (SIGGRAPH 17)')
# plt.axis('off')
# plt.show()

#
# # Load the MNIST dataset and apply transformations
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#
# # Initialize the model and specify a loss function and optimizer
# net = Net()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
# # Training loop
# for epoch in range(5):  # Number of epochs
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")
#
# print("Finished Training")
