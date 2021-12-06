import os
import numpy as np
from dataloader import *
from model import *
from torch.utils.data import Dataset, DataLoader
from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cuda:0",
                    help='test time gpu device id')
parser.add_argument('--dataset', type=str, default='A2C',
                    help='pascal or cityscapes')
parser.add_argument('--workers', type=int, default=1,
                    help='number of data loading workers')
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--parameter', type=str)
args = parser.parse_args()


def main():
    device = torch.device(args.device)
    if args.dataset == "A2C":
        test_datapath = "/home/nas_datasets/hyoungwoodata/validation/A2C"
        test_dataset = MyDataset(test_datapath)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
        
    elif args.dataset == "A4C":
        test_datapath = "/home/nas_datasets/hyoungwoodata/validation/A4C"
        test_dataset = MyDataset(test_datapath)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    
    if args.model == "unet":
        model = UNet(n_class=1)
    elif args.model == "resnet50":
        model = resnet50(num_classes=1)
    elif args.model == "resnet101":
        model = resnet101(num_classes=1)
    elif args.model == "resnet152":
        model = resnet152(num_classes=1)
        
    checkpoint = torch.load(args.parameter, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device)
    
    losses = AverageMeter()
    
    DSC = DiceLoss()
    JI = JaccardIndex()
    dice_score = 0
    jaccard_score = 0
    
    for i, sample in enumerate(test_dataloader):
        with torch.no_grad():
            model.eval()
            img = sample["image"].to(device)
            label = sample["mask"].to(device)
            
            output = model(img)
            
            dice = DSC(output, label)
            jaccard_index = JI(output, label)
            
            dice_score += dice
            jaccard_score += jaccard_index
            
    print(f"{args.dataset} {args.model} DSC {dice_score / len(test_dataloader)}, JI {jaccard_score / len(test_dataloader)}") 
    

if __name__ == "__main__":
    main()