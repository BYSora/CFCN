import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
# from u_net import unet
from model import unet
from CFCN import cfcn
from util import FocalLoss, DetDataset, ClsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

import os


def get_train_transforms():
    train_transform = [
        A.Flip(always_apply=False, p=0.5),
        A.Transpose(always_apply=False, p=0.5),
        A.Rotate(limit=90, interpolation=1, border_mode=4, always_apply=False, p=0.5),
        A.ShiftScaleRotate(),
        A.RGBShift(),
        A.GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=0.5)
    ]
    return A.Compose(train_transform)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if not os.path.exists("weights/"):
        os.makedirs("weights/")

    # model = cfcn()
    model = unet(1)
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    batch_size = 16
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    det_dataset = ClsDataset('train', augmentation=get_train_transforms())
    data_loaders = DataLoader(det_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    val_dataset = ClsDataset('val')
    val_loaders = DataLoader(val_dataset, batch_size=8)
    epoch_num = 150

    best_loss = 99999.0
    el = []
    vl = []
    for epoch in range(epoch_num):
        print('Epoch {}/{}'.format(epoch + 1, epoch_num))
        model.train()
        epoch_loss = 0
        step = 0
        for x, y, _ in tqdm(data_loaders):
            step += 1
            inputs = x.cuda()
            labels = y.cuda()
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, dim=1)
            labels = torch.squeeze(labels, dim=1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = 0.0
        with torch.no_grad():
            model.eval()
        with torch.no_grad():
            for x, y, _ in val_loaders:
                inputs = x.cuda()
                labels = y.cuda()
                outputs = model(inputs)
                outputs = torch.squeeze(outputs, dim=1)
                labels = torch.squeeze(labels, dim=1)
                vloss = criterion(outputs, labels)
                val_loss += vloss.item()
            if val_loss < best_loss:
                best_loss = val_loss
                for file in os.listdir('weights/'):
                    if 'best' in file:
                        os.remove(os.path.join('weights/', file))
                torch.save(model.module.state_dict(), 'weights/weights_best_{}_{}.pth'.format(epoch+1, round(val_loss, 3)))
                torch.save(model.module.state_dict(), 'weights/weights_best.pth')

        print("train_loss:%f   val_loss:%f" % (epoch_loss, val_loss))
        print('-' * 10)
        el.append(epoch_loss)
        vl.append(val_loss)

    torch.save(model.module.state_dict(), 'weights/weights_last.pth')
    el[0] = 0
    plt.plot(range(epoch_num), el)
    vl[0] = 0
    plt.plot(range(epoch_num), vl)
    plt.savefig('result.png')


if __name__ == '__main__':
    train()
