import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from utils.tool import deblur_image
class L10(nn.Module):
    """
    输入： (batch, 3, H, W)  值域[-0.5, 0.5]
    输出： (batch, 3, H, W)
    """
    def __init__(self):
        super(L10, self).__init__()
        #利用padding保证输出大小不变
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)          # 7x7
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 160, kernel_size=1)                   # 1x1
        self.conv3 = nn.Conv2d(160, 160, kernel_size=3, padding=1)        # 3x3
        self.conv4 = nn.Conv2d(160, 64, kernel_size=1)                   # 1x1
        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, padding=2)        # 5x5
        self.conv6 = nn.Conv2d(128, 256, kernel_size=5, padding=2)       # 5x5
        self.conv7 = nn.Conv2d(256, 128, kernel_size=1)                  # 1x1
        self.conv8 = nn.Conv2d(128, 32, kernel_size=7, padding=3)       # 7x7
        self.conv9 = nn.Conv2d(32, 32, kernel_size=3, padding=1)                    # 1x1
        self.conv10 = nn.Conv2d(32, 3, kernel_size=1) 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.conv10(x)         
        return x
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model= L10().to(device)
model.load_state_dict(torch.load('checkpoints/run_20260224_161238_L10_bs/checkpoint_epoch100.pth', map_location=device))
print("模型已加载")

input_image = r'D:\CNN_for_Direct_Text_Deblurring\Code\dataset\val\blurred\000002.png'                            
output_image = deblur_image(model, input_image, device)


output_path = 'deblurred_result.png'
cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
print(f"去模糊图像已保存至: {output_path}")