import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda")

files_dir = "./lgg-mri-segmentation/kaggle_3m/"

'''
* 代表着在该目录下的子目录
[0-9].tif 是对文件名的限定，要求文件名以数字（0 - 9）结尾，并且文件扩展名为 .tif。
也就是获取img路径
'''
file_paths = glob(f'{files_dir}/*/*[0-9].tif')

csv_path = './lgg-mri-segmentation/kaggle_3m/data.csv'
df = pd.read_csv(csv_path)


#SimpleImputer(strategy="most_frequent")创建了一个填补器对象，它会用出现频率最高的值（众数）来填补缺失值。
imputer = SimpleImputer(strategy="most_frequent")
'''
imputer.fit_transform(df)对数据框df进行拟合和转换：
fit方法会统计每列的众数。
transform方法会用计算得到的众数替换每列中的缺失值（NaN）。
该操作返回的是一个 numpy 数组，随后这个数组会被重新转换为 DataFrame。
columns=df.columns确保重建的 DataFrame 保留原始的列名。
'''
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 传入的path路径代表img路径，后面都是基于此操作的
def get_file_row(path):
    # os.path.splitext(path)：该操作会把路径分割成文件名和扩展名两部分，为得到对应的mask做准备
    path_no_ext, ext = os.path.splitext(path)
    # os.path.basename(path)：此操作会提取路径中的文件名部分，就是最后一级的那个名字
    filename = os.path.basename(path)
    # 病人的id由三部分组成（唯一确定，最后一部分不要）
    # '_'.join(...)：取列表的前三个元素，再用_连接起来，从而形成患者 ID
    patient_id = '_'.join(filename.split('_')[:3])
    # 然后这里返回的还包括对应的mask，应为他和img的区别仅仅是多了_mask
    return [patient_id, path, f'{path_no_ext}_mask{ext}']

# 得到我们最终的一个dataframe
filenames_df = pd.DataFrame((get_file_row(filename) for filename in file_paths),
                            columns=['Patient', 'image_filename', 'mask_filename'])


# 定义dataset，数据预处理
class MriDataset(Dataset):
    def __init__(self, df, transform=None):
        super(MriDataset, self).__init__()
        self.df = df
        self.transform = transform
        # 先实例化
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, raw=False):
        # iloc获取行数据
        row = self.df.iloc[idx]
        # OpenCV 库中用于读取图像文件的函数，返回 NumPy 数组
        img = cv2.imread(row['image_filename'], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(row['mask_filename'], cv2.IMREAD_GRAYSCALE)
        if raw:
            return img, mask

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # cv2读取的时候是以BGR顺序读取的，需要先转换通道顺序
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.to_tensor(img)
        # 读取的掩码像素值为 0 或 255，通过 mask // 255 将其归一化为 0 或 1
        mask = mask // 255
        mask = torch.Tensor(mask)
        return img, mask
#拼接
df = pd.merge(df, filenames_df, on="Patient")

#分割数据集：训练，验证，测试
train_df, test_df = train_test_split(df, test_size=0.3)
test_df, valid_df = train_test_split(test_df, test_size=0.5)

#数据增强
transform = A.Compose([
    A.ChannelDropout(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.ColorJitter(p=0.3),
])

#得到dataset
train_dataset = MriDataset(train_df, transform)
valid_dataset = MriDataset(valid_df)
test_dataset = MriDataset(test_df)
batch_size = 16

#得到dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

if __name__ == '__main__':
    # 设置要显示的样本数量
    n_examples = 4

    # 创建子图布局：n_examples行，3列，设置图像大小和布局
    fig, axs = plt.subplots(n_examples, 3, figsize=(20, n_examples * 7), constrained_layout=True)

    # 初始化索引
    i = 0

    # 遍历每个样本的子图行
    for ax in axs:
        while True:
            # 从训练数据集中获取图像和掩码，raw=True表示获取原始数据
            image, mask = train_dataset.__getitem__(i, raw=True)
            i += 1

            # 检查掩码中是否存在异常区域（如果全为0则表示无异常）
            if np.any(mask):
                # 在第一列显示原始MRI图像
                ax[0].set_title("MRI images")
                ax[0].imshow(image)

                # 在第二列显示高亮异常区域的图像（原图+半透明掩码）
                ax[1].set_title("Highlited abnormality")
                ax[1].imshow(image)
                ax[1].imshow(mask, alpha=0.2)

                # 在第三列显示纯掩码图像
                ax[2].imshow(mask)
                ax[2].set_title("Abnormality mask")
                break
    # 保存图像
    plt.savefig('segmentation_results.png', dpi=300, bbox_inches='tight')









