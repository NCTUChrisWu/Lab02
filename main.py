import copy
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from models.EEGNet import EEGNet
from models.EEGNet import DeepConvNet
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
import os

class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]

def plot_train_acc(train_acc_list, epochs):
    os.makedirs('./figures', exist_ok=True)
    x = list(range(1, len(train_acc_list) + 1))
    plt.figure()
    plt.plot(x, train_acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig('./figures/train_acc.png', dpi=200)
    plt.close()

def plot_train_loss(train_loss_list, epochs):
    os.makedirs('./figures', exist_ok=True)
    x = list(range(1, len(train_loss_list) + 1))
    plt.figure()
    plt.plot(x, train_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig('./figures/train_loss.png', dpi=200)
    plt.close()

def plot_test_acc(test_acc_list, epochs):
    os.makedirs('./figures', exist_ok=True)
    x = list(range(1, len(test_acc_list) + 1))
    plt.figure()
    plt.plot(x, test_acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Testing Accuracy')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig('./figures/test_acc.png', dpi=200)
    plt.close()

    

def train(model, loader, criterion, optimizer, args,
          start_epoch=0,
          avg_acc_list=None,
          avg_loss_list=None,
          test_acc_list=None):

    if avg_acc_list is None: avg_acc_list = []
    if avg_loss_list is None: avg_loss_list = []
    if test_acc_list is None: test_acc_list = []

    best_acc = max(test_acc_list) if len(test_acc_list) > 0 else 0.0
    best_epoch = -1 if len(test_acc_list) == 0 else test_acc_list.index(best_acc) + 1
    best_wts = model.state_dict()


    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0
            for i, data in enumerate(tqdm(loader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)

            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)

            print(f'Epoch: {epoch + 1}')
            print(f'Loss: {avg_loss}')
            print(f'Training Acc. (%): {avg_acc:3.2f}%')

        # keep test() untouched
        test_acc = test(model, test_loader)
        test_acc_list.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_wts = model.state_dict()

            print("New BEST model found!")
            print(f"    Best Epoch : {best_epoch}")
            print(f"    Best Acc   : {best_acc:.2f}%")

        print(f'Test Acc. (%): {test_acc:3.2f}%')

        os.makedirs('./weights', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_acc_list': avg_acc_list,
            'avg_loss_list': avg_loss_list,
            'test_acc_list': test_acc_list,
            'best_acc': best_acc,
        }, './weights/ckpt.pt')

    torch.save(best_wts, './weights/best.pt')
    print("====================================")
    print("Training Finished")
    print(f"Best Epoch : {best_epoch}")
    print(f"Best Test Acc : {best_acc:.2f}%")
    print("====================================")
    return avg_acc_list, avg_loss_list, test_acc_list



def test(model, loader):
    avg_acc = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=150)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true") ##

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # TODO write EEGNet yourself
    # model = EEGNet()
    model = DeepConvNet(num_classes=2, chans=2, samples=750, dropout=0.5)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    model.to(device)
    criterion.to(device)

    start_epoch = 0
    train_acc_list, train_loss_list, test_acc_list = [], [], []

    if args.resume and os.path.exists('./weights/ckpt.pt'):
        ckpt = torch.load('./weights/ckpt.pt', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        train_acc_list = ckpt['avg_acc_list']
        train_loss_list = ckpt['avg_loss_list']
        test_acc_list = ckpt['test_acc_list']
        print(f"Resume from epoch {start_epoch + 1}")


    train_acc_list, train_loss_list, test_acc_list = train(
        model,
        train_loader,
        criterion,
        optimizer,
        args,
        start_epoch=start_epoch,
        avg_acc_list=train_acc_list,
        avg_loss_list=train_loss_list,
        test_acc_list=test_acc_list
    )
    plot_train_acc(train_acc_list, args.num_epochs)
    plot_train_loss(train_loss_list, args.num_epochs)
    plot_test_acc(test_acc_list, args.num_epochs)
