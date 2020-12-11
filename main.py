# Usage example

from MyRandomResizeTransform import MyRandomResizeTransform
from dataset import MyDataset
from torch.utils.data import DataLoader, random_split
from dataset.my_data_loader import MyDataLoader


image_size = [192, 256, 320, 384, 448]

MyRandomResizedCrop.IMAGE_SIZE_LIST = image_size.copy()
MyRandomResizedCrop.ACTIVE_SIZE = max(image_size)
train_transforms = MyRandomResizedCrop(size=0)
train_list = 'dataset/train.txt'
dataset = MyDataset(train_list, train_transforms)
loader = MyDataLoader(dataset, batch_size=5, num_workers=8, pin_memory=True, shuffle=True)

for idx, batch in enumerate(loader):
    # print(batch)
    print(batch['image'].shape)
    print(batch['mask'].shape)
    print(batch['class'])
    print()
    if idx==5:
        break