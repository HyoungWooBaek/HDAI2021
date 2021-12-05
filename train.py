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
parser.add_argument('--epochs', type=int, default=300,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.00025,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--resume', type=str, default="/home/nas_datasets/hyoungwoodata/checkpoint",
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=1,
                    help='number of data loading workers')
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--loss', type=str, default='bce')
args = parser.parse_args()


def main():
    device = torch.device(args.device)
    if args.dataset == "A2C":
        train_datapath = "/home/nas_datasets/hyoungwoodata/train/A2C"
        train_dataset = MyDataset(train_datapath)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        
        val_datapath = "/home/nas_datasets/hyoungwoodata/validation/A2C"
        val_dataset = MyDataset(val_datapath)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
    elif args.dataset == "A4C":
        train_datapath = "/home/nas_datasets/hyoungwoodata/train/A4C"
        train_dataset = MyDataset(train_datapath)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        
        val_datapath = "/home/nas_datasets/hyoungwoodata/validation/A4C"
        val_dataset = MyDataset(val_datapath)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
    else:
        train_a2c_path = "/home/nas_datasets/hyoungwoodata/train/A2C"
        train_a4c_path = "/home/nas_datasets/hyoungwoodata/train/A4C"
        train_dataset = AllDataset(train_a2c_path, train_a4c_path)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        
        val_a2c_path = "/home/nas_datasets/hyoungwoodata/validation/A2C"
        val_a4c_path = "/home/nas_datasets/hyoungwoodata/validation/A4C"
        val_dataset = AllDataset(val_a2c_path, val_a4c_path)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
    if args.model == "unet":
        model = UNet(n_class=1)
        
    model.to(device)
    
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    max_iter = args.epochs * len(train_dataloader)
    losses = AverageMeter()
    
    DSC = DiceLoss()
    JI = JaccardIndex()
    best_score = 0
    
    for epoch in range(args.epochs):
        for i, sample in enumerate(train_dataloader):
            model.train()
            cur_iter = epoch * len(train_dataloader) + i
            lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
            model.train()
            img = sample["image"].to(device)
            label = sample["mask"].to(device)
            label = label.unsqueeze(1)
            
            output = model(img)
            if args.loss == 'bce':
                loss = bce_loss(output, label)
            elif args.loss == 'dice':
                loss, _ = dice_loss(output, label)
            
            losses.update(loss.item(), args.batch_size)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i % 50 == 0:
                print('epoch: {0}\t'
                      'iter: {1}/{2}\t'
                      'lr: {3:.6f}\t'
                      'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                          epoch + 1, i + 1, len(train_dataloader), lr, loss=losses))
                
            
                
        score = 0
        for i, sample in enumerate(val_dataloader):
            with torch.no_grad():
                model.eval()
                img = sample["image"].to(device)
                label = sample["mask"].to(device)
                
                output = model(img)
                
                dice = DSC(output, label)
                jaccard_index = JI(output, label)
                
                score += dice + jaccard_index
                
            
        if score >= best_score:
            best_score = score
            torch.save(model.state_dict(), f"{args.resume}/best.pth")
            print(f"epoch: {epoch + 1} best score: {best_score.item()}")
            
        

if __name__ == "__main__":
    main()