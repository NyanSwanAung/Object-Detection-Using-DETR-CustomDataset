# Object-Detection-Using-DETR-CustomDataset
Training DETR on Custom Dataset for Object Detection

## Face Dataset
In [dataloader/face.py](https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/blob/main/dataloaders/face.py), I set the maximum width of images in the random transform to 800 pixels. This should allow for training on most GPUs, but it is advisable to change back to the original 1333 if your GPU can handle it.

## Augmentation methods 
For train images, 
``` 
T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=800),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=800),
                ])
```

For val images, 

``` T.RandomResize([800], max_size=800) ```
