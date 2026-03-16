from module import rx50
from my_dataset import device, test_loader, test_dataset
import torch
from train import iou_pytorch, dice_pytorch, BCE_dice
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

def text_module():
    rx50.load_state_dict(torch.load("weights.pt"))
    rx50.eval()
    loss_fn = BCE_dice
    with torch.no_grad():
        running_IoU = 0
        running_dice = 0
        running_loss = 0
        for i, data in enumerate(test_loader):
            img, mask = data
            img, mask = img.to(device), mask.to(device)
            predictions = rx50(img)
            predictions = predictions.squeeze(1)
            running_dice += dice_pytorch(predictions, mask).sum().item()
            running_IoU += iou_pytorch(predictions, mask).sum().item()
            loss = loss_fn(predictions, mask)
            running_loss += loss.item() * img.size(0)
        loss = running_loss / len(test_dataset)
        dice = running_dice / len(test_dataset)
        IoU = running_IoU / len(test_dataset)
        print(f'Tests: loss: {loss} | Mean IoU: {IoU} | Dice coefficient: {dice}')

    #可视化检测结果
    width = 3
    columns = 10
    n_examples = columns * width

    fig, axs = plt.subplots(columns, width, figsize=(7 * width, 7 * columns), constrained_layout=True)
    red_patch = mpatches.Patch(color='red', label='The red data')
    fig.legend(loc='upper right', handles=[
        mpatches.Patch(color='red', label='Ground truth'),
        mpatches.Patch(color='green', label='Predicted abnormality')])
    i = 0
    with torch.no_grad():
        for data in test_loader:
            image, mask = data
            mask = mask[0]
            if not mask.byte().any():
                continue
            image = image.to(device)
            prediction = rx50(image).to('cpu')[0][0]
            prediction = torch.where(prediction > 0.5, 1, 0)
            prediction_edges = prediction - binary_dilation(prediction)
            ground_truth = mask - binary_dilation(mask)
            image[0, 0, ground_truth.bool()] = 1
            image[0, 1, prediction_edges.bool()] = 1

            axs[i // width][i % width].imshow(image[0].to('cpu').permute(1, 2, 0))
            if n_examples == i + 1:
                break
            i += 1
    # 保存图像
    plt.savefig('segmentation_results.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    text_module()