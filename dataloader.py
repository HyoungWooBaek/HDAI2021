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
        mask2 = np.array(mask)
        
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
        img, mask = self.tensor(img), torch.FloatTensor(np.array(mask))
        return {'image': img, 'mask': mask}
        
        
class MyDataset(Dataset):
    def __init__(self, root_dir=None):
        path = os.path.join(root_dir)
            
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
        
        # img = Image.fromarray(255*label[0].numpy())
        # img.show()
        sample = {'image': _img, 'mask': _label}
        sample = self.transform(sample)

        return sample
    
    
class AllDataset(Dataset):
    def __init__(self, a2c_root_dir=None, a4c_root_dir=None):
        a2c_path = a2c_root_dir
        a4c_path = a4c_root_dir
            
        a2c_file_list = [os.path.join(a2c_path, file) for file in os.listdir(a2c_path)]
        a4c_file_list = [os.path.join(a4c_path, file) for file in os.listdir(a4c_path)]
        
        a2c_image_list = [file for file in a2c_file_list if ".png" in file]
        a2c_label_list = [file for file in a2c_file_list if ".npy" in file]
        
        a4c_image_list = [file for file in a4c_file_list if ".png" in file]
        a4c_label_list = [file for file in a4c_file_list if ".npy" in file]
        
        self.all_image_list = []
        self.all_image_list.extend(a2c_image_list)
        self.all_image_list.extend(a4c_image_list)
        
        self.all_label_list = []
        self.all_label_list.extend(a2c_label_list)
        self.all_label_list.extend(a4c_label_list)
        
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(224),
             ToTensor()])
            
    def __len__(self):
        return len(self.all_image_list)

    def __getitem__(self, idx):
        _img = Image.open(self.all_image_list[idx]).convert('L')
        _label = np.load(self.all_label_list[idx])
        _label = Image.fromarray(_label)
        
        # img = Image.fromarray(255*label[0].numpy())
        # img.show()
        sample = {'image': _img, 'mask': _label}
        sample = self.transform(sample)
        
        return sample