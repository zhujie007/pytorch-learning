from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from module import rx50
from my_dataset import device, train_loader, valid_loader
from torch.utils.tensorboard import SummaryWriter
#设置早停机制
class EarlyStopping():
    def __init__(self, patience=6, min_delta=0, weights_path='weights.pt'):
        self.patience = patience  # 允许损失不改善的最大轮数
        self.min_delta = min_delta  # 视为"改善"所需的最小损失差异
        self.counter = 0  # 记录损失未改善的轮数
        self.best_loss = float('inf')  # 初始最佳损失设为无穷大
        self.weights_path = weights_path  # 保存最佳模型权重的路径

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if self.best_loss - val_loss > self.min_delta:
            # 若当前损失显著低于最佳损失，更新最佳损失并保存模型
            self.best_loss = val_loss
            # 保存方式2，把模型参数保存为字典，没有结构，占用空间小
            torch.save(model.state_dict(), self.weights_path)
            self.counter = 0  # 重置计数器
        else:
            # 若损失未改善，增加计数器
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 触发早停
        return False  # 继续训练
#加载模型权重
    def load_weights(self, model: torch.nn.Module):
        # 从self.weights_path指定的文件中加载模型状态字典
        # 并将其应用到传入的model上
        return model.load_state_dict(torch.load(self.weights_path))


# 评价指标
# iou=交集/并集
def iou_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    predictions = torch.where(predictions > 0.5, 1, 0)  # 二值化预测结果
    labels = labels.byte()  # 转换为byte类型

    intersection = (predictions & labels).float().sum((1, 2))  # 计算交集
    union = (predictions | labels).float().sum((1, 2))  # 计算并集

    iou = (intersection + e) / (union + e)  # 计算IoU，添加epsilon防止除零
    return iou

# Dice ≈ 2×IoU / (IoU + 1)，两者均衡量分割重叠程度，但 Dice 对小目标更敏感。
def dice_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    predictions = torch.where(predictions > 0.5, 1, 0)  # 二值化预测结果
    labels = labels.byte()  # 转换为byte类型
    intersection = (predictions & labels).float().sum((1, 2))  # 计算交集
    return ((2 * intersection) + e) / (predictions.float().sum((1, 2)) + labels.float().sum((1, 2)) + e)
#结合 BCE 和 Dice 的优点，alpha控制 Dice 损失的权重
def BCE_dice(output, target, alpha=0.01):
    bce = torch.nn.functional.binary_cross_entropy(output, target)  # 计算BCE损失
    soft_dice = 1 - dice_pytorch(output, target).mean()  # 计算Dice损失（1-Dice）
    return bce + alpha * soft_dice  # 加权组合

#训练脚本
def training(epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler):
    # 使用tensorboard进行可视化
    writer = SummaryWriter(log_dir='./logs')
    early_stopping = EarlyStopping(patience=7)

    for epoch in range(1, epochs + 1):

        running_loss = 0
        #如果含有dropout，这个还有.eval()得写上
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            img, mask = data
            img, mask = img.to(device), mask.to(device)
            predictions = model(img)
            predictions = predictions.squeeze(1)
            loss = loss_fn(predictions, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img.size(0)
        # 计算训练损失并记录到TensorBoard
        train_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar(f'Train_Loss', train_loss, epoch)

        model.eval()
        running_IoU = 0
        running_dice = 0
        running_valid_loss = 0
        with torch.no_grad():
            for data in valid_loader:
                img, mask = data
                img, mask = img.to(device), mask.to(device)
                predictions = model(img)
                predictions = predictions.squeeze(1)
                running_dice += dice_pytorch(predictions, mask).sum().item()
                running_IoU += iou_pytorch(predictions, mask).sum().item()
                loss = loss_fn(predictions, mask)
                running_valid_loss += loss.item() * img.size(0)


        val_loss = running_valid_loss / len(valid_loader.dataset)
        val_dice = running_dice / len(valid_loader.dataset)
        val_IoU = running_IoU / len(valid_loader.dataset)

        #记录每个epoch验证集指标变化
        writer.add_scalar('Val_Loss', val_loss, epoch)
        writer.add_scalar('Val_IoU', val_IoU, epoch)
        writer.add_scalar('Val_Dice', val_dice, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        writer.close()

        lr_scheduler.step(val_loss)
        if early_stopping(val_loss, model):
            #返回最优模型
            early_stopping.load_weights(model)
            break


if __name__ == '__main__':
    #实例化损失函数
    loss_fn = BCE_dice
    #使用adam优化器
    optimizer = Adam(rx50.parameters(), lr=5e-4)
    epochs = 60
    lr_scheduler = ReduceLROnPlateau(
        optimizer=optimizer,  # 指定要调整学习率的优化器
        patience=2,  # 容忍多少个epoch没有改善
        factor=0.2  # 学习率降低的因子 (new_lr = old_lr * factor)
    )
    training(epochs, rx50, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler)