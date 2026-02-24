import torch
from datetime import datetime
from torch.utils import data as Data
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os

class CenterMSELoss(nn.Module):
    """
    计算输出图像中心 16x16 区域与目标清晰块的 MSE。
    """
    def __init__(self, target_size=16):
        super(CenterMSELoss, self).__init__()
        self.target_size = target_size

    def forward(self, output, target):
        _, _, h, w = output.shape
        start = (h - self.target_size) // 2
        output_center = output[:, :, start:start+self.target_size, start:start+self.target_size]
        return nn.functional.mse_loss(output_center, target)
def train(model, train_loader, val_loader, epochs, lr=1e-4, device='cuda'):
    """"
    训练函数,优化器为SGD,损失函数为CenterMSELoss,每10轮保存一次模型权重
    """
    # 创建带时间戳的保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'run_{timestamp}_L15_bs'
    checkpoint_dir = os.path.join('checkpoints', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f'All checkpoints will be saved under: {checkpoint_dir}')
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = CenterMSELoss(target_size=16)
    model.to(device)

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for blur, target in tqdm(train_loader, desc=f'Epoch {epoch}'):
            blur, target = blur.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(blur)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * blur.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for blur, target in val_loader:
                blur, target = blur.to(device), target.to(device)
                output = model(blur)
                loss = criterion(output, target)
                val_loss += loss.item() * blur.size(0)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        if epoch % 10 == 0:
            filename = f'checkpoint_epoch{epoch}.pth'
            filepath = os.path.join(checkpoint_dir, filename)
            torch.save(model.state_dict(), filepath)
