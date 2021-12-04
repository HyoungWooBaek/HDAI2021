import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = img.resize((256, 256), resample=Image.BILINEAR), mask.resize((256, 256), resample=Image.BILINEAR)
        h, w = img.size
        new_h, new_w = self.size, self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img = img.crop((left, top, left + new_w, top + new_h))
        mask = mask.crop((left, top, left + new_w, top + new_h))

        return {'image': img, 'mask': mask}


class RandomFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self.flip = transforms.RandomHorizontalFlip(1.)

    def __call__(self, sample):
        if np.random.random_sample() < self.prob:
            img, mask = sample['image'], sample['mask']
            img = self.flip(img)
            mask = self.flip(mask)
            return {'image': img, 'mask': mask}
        else:
            return sample
        
        
class ToTensor(object):
    def __init__(self):
        self.tensor = transforms.ToTensor()

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = self.tensor(img), self.tensor(mask)
        return {'image': img, 'mask': mask}
        
        
class MyDataset(Dataset):
    def __init__(self, root_dir=None, istrain=True):
        path = os.path.join(root_dir, 'train/A2C')
            
        file_list = [os.path.join(path, file) for file in os.listdir(path)]
        self.image_list = [file for file in file_list if ".png" in file]
        self.label_list = [file for file in file_list if ".npy" in file]
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(224),
             ToTensor()])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        _img = Image.open(self.image_list[idx]).convert('L')
        _label = np.load(self.label_list[idx])
        _label = Image.fromarray(_label)
        
        print(_img.size, self.image_list[idx])
        # img = Image.fromarray(255*label[0].numpy())
        # img.show()
        sample = {'image': _img, 'mask': _label}
        sample = self.transform(sample)

        return sample