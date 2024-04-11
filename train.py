import torch 
import torch.nn as nn 
from tqdm import tqdm 
import argparse 
import numpy as np 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.optim import lr_scheduler
from dataset import Monkey
from model import *
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

torch.autograd.set_detect_anomaly(True)

def train(args, model, train_loader, val_loader, optimizer, device): 
    
    criterion = nn.MSELoss()
    best_val_loss = 1e9
    for e in range(1000):
        train_loss = 0.
        model.train()
        for idx, data in tqdm(enumerate(train_loader)):
            data = data.to(device) # [Batch, Window, Channel, Input_dim]
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred[:, -1, :, :], data[:, -1, :, :])
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss = train_loss + loss.item()
        print(f"training_loss at epoch {e+1}: {train_loss / len(train_loader)}")
        model.eval()
        validation_loss = 0.
        with torch.no_grad():
            for idx, data in tqdm(enumerate(val_loader)):
                data = data.to(device) # [Batch, Window, Channel, Input_dim]
                optimizer.zero_grad()
                pred = model(data)
                loss = criterion(pred[:, -1, :, :], data[:, -1, :, :])
                validation_loss = validation_loss + loss.item()
            print(f"validation_loss at epoch {e+1}: {validation_loss / len(val_loader)}")
            if best_val_loss > validation_loss / len(val_loader):
                best_val_loss = validation_loss / len(val_loader)
                torch.save(model.state_dict(), './storage/ckpt/amag_best.pt')
    torch.save(model.state_dict(), f"./storage/ckpt/amag_final_loss_{validation_loss/len(val_loader)}.pt")
        
            

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='beignet') 
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-l', '--lr', type=float, default=0.0005)
    parser.add_argument('-e', '--max_epoch', type=int, default=200) 
    parser.add_argument('--num_propagate', type=int, default=1)
    parser.add_argument('--device', type=str, default='gpu')
    args = parser.parse_args()
    if args.device == 'gpu':
        device = torch.device('cuda')
    else: 
        device = torch.device('cpu')
    training_data = Monkey(args.dataset, split='train')
    val_data = Monkey(args.dataset, split='val')
    corr = Monkey.calculate_corr_matrix(training_data.data)
    model = AMAG_transformer(9, 9, args.hidden_size, num_channels=corr.shape[0], 
                 num_propagate=args.num_propagate, 
                 mode='train', corr=corr, device=device)
    
    model = nn.DataParallel(model).to(device)
    # model = model.to(device)
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, train_loader, val_loader, optimizer, device)
    

    