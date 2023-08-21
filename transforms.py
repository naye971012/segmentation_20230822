import albumentations as A
from albumentations.pytorch import ToTensorV2

test_transform = A.Compose([
        A.Resize(1080, 1920),
        A.Normalize(),
        ToTensorV2()
])

train_transform = A.Compose([
        A.Resize(540, 960),
        A.Normalize(),
        ToTensorV2()
])