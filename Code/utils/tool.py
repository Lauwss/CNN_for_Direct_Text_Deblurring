import torch
from datetime import datetime
from torch.utils import data as Data
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    criterion = CenterMSELoss(target_size=16)
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs+1):
        model.train()
        total_train_loss = 0.0
        for blur, target in tqdm(train_loader, desc=f'Epoch {epoch}'):
            blur, target = blur.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(blur)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * blur.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for blur, target in val_loader:
                blur, target = blur.to(device), target.to(device)
                output = model(blur)
                loss = criterion(output, target)
                total_val_loss += loss.item() * blur.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        scheduler.step()
        print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
        if epoch % 10 == 0:
            filename = f'checkpoint_epoch{epoch}.pth'
            filepath = os.path.join(checkpoint_dir, filename)
            torch.save(model.state_dict(), filepath)
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Curve")
    plt.show()