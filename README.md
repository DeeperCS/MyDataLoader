## Steps for generating batches with random sized images

### Notice:
##### The same class MyRandomResizeTransform must be imported in my_data_loader.py
##### Since the variables IMAGE_SIZE_LIST and ACTIVE_SIZE are shared at class level
```
from dataset.MyRandomResizeTransform import MyRandomResizeTransform
```

### 1. Transform
##### Constructs a transform class for resizing images
```
from dataset.MyRandomResizeTransform import MyRandomResizeTransform
image_size = [160, 192, 224, 256]
image_size.sort()  # e.g., 160 -> 224
MyRandomResizeTransform.IMAGE_SIZE_LIST = image_size.copy()
MyRandomResizeTransform.ACTIVE_SIZE = max(image_size)

train_transforms = MyRandomResizeTransform(size=0)
```


### 2. Dataset 
##### Applies the train_transforms to each of the loaded image
```
from dataset.dataset import MyDatasetTransform

train_list = 'dataset/train.txt'
train_dataset = MyDatasetTransform(train_list, train_transforms)
```


### 3. DataLoader
##### Random set ACTIVE_SIZE for each batch

```
from dataset.my_data_loader import MyDataLoader
train = MyDataLoader(train_dataset, batch_size=5, num_workers=4, pin_memory=True)
```

### 4. Use it
##### Now every batch is in different size
```
for idx, batch in enumerate(train):
    # print(batch)
    print(batch['image'].shape)
    if idx==10:
        break
  
  
output:
torch.Size([5, 3, 192, 192])
torch.Size([5, 3, 160, 160])
torch.Size([5, 3, 160, 160])
torch.Size([5, 3, 192, 192])
torch.Size([5, 3, 160, 160])
torch.Size([5, 3, 256, 256])
torch.Size([5, 3, 192, 192])
torch.Size([5, 3, 256, 256])
torch.Size([5, 3, 256, 256])
torch.Size([5, 3, 192, 192])
torch.Size([5, 3, 160, 160])
```


Test env:
```
torch.__version__: '1.7.0+cu101'
torchvision.__version__: '0.8.1+cu101'
PIL.__version__:'5.3.0'
```

[Reference]

[Once for All](https://github.com/mit-han-lab/once-for-all)