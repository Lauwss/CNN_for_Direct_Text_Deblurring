import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
from utils.tool import CenterMSELoss, train

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


class DeblurDataset(Dataset):
    """
    从文件夹加载 (模糊块, 目标清晰块) 图像对。
    """
    def __init__(self, root, split='train'):
        self.blur_dir = os.path.join(root, split, 'blurred')
        self.target_dir = os.path.join(root, split, 'target')
        self.filenames = sorted([f for f in os.listdir(self.blur_dir) if f.endswith('.png')])
        assert len(self.filenames) == len(os.listdir(self.target_dir)), "Mismatch between blurred and target files"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        # 读取图像并转为 RGB
        blur_path = os.path.join(self.blur_dir, fname)
        target_path = os.path.join(self.target_dir, fname)
        blur_img = cv2.imread(blur_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        target_img = cv2.imread(target_path)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # 转为 float32 并归一化到 [0,1]
        blur_img = blur_img.astype(np.float32) / 255.0
        target_img = target_img.astype(np.float32) / 255.0

        # 网络要求输入输出值域为 [-0.5, 0.5]
        blur_img = blur_img - 0.5
        target_img = target_img - 0.5

        # 转换为 tensor (C, H, W)
        blur_tensor = torch.from_numpy(blur_img).permute(2, 0, 1)
        target_tensor = torch.from_numpy(target_img).permute(2, 0, 1)
        return blur_tensor, target_tensor
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def main():
    data_root = 'Code\dataset'         
    batch_size = 32                
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    train_dataset = DeblurDataset(data_root, split='train')
    val_dataset = DeblurDataset(data_root, split='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=0)
    # for x in train_loader:
    #  print(x[0].shape)
    model = L10()
    model.apply(init_weights)
    train(model, train_loader, val_loader, epochs, lr=1e-3, device=device)

if __name__ == '__main__':
    main()