import torch
import torch.nn as nn
import random
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import struct
import numpy as np
from tqdm import  tqdm
import torchvision

def load_labels(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)


def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, 28,28)


class GanDataset(Dataset):
    def __init__(self,imgs,labels):
        self.imgs = imgs
        self.labels = labels


    def __getitem__(self, index):
        img = self.imgs[index]
        lab = self.labels[index]

        img = img.astype(np.float32)
        lab = lab.astype(np.int32)

        return img,lab


    def __len__(self):
        return len(self.imgs)


class Dis_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784,2000)
        self.lrelu = nn.LeakyReLU()
        self.linear2 = nn.Linear(2000,1)

    def forward(self,x):
        x = self.linear1(x)
        x = self.lrelu(x)
        x = self.linear2(x)

        return x


class Gen_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(512,1024)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024,784)


    def forward(self,batch_size): # batch_size = 10
        x = torch.randn(size=(batch_size,512),device=device)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # x = x.reshape(batch_size,28,28)

        return x

def save_img(imgs,e):
    imgs = imgs.reshape(imgs.shape[0],1,28,28)
    imgs = imgs.clamp(0,1)

    imgs = torchvision.utils.make_grid(imgs,nrow=8).detach().cpu().numpy()
    plt.imshow(imgs.transpose(1,2,0))
    plt.savefig(f"result2\\{e}.jpg")



if __name__ == "__main__":
    train_imgs = load_images("data\\train-images-idx3-ubyte") / 255
    train_labs = load_labels("data\\train-labels-idx1-ubyte")

    test_imgs = load_images("data\\t10k-images-idx3-ubyte") / 255
    test_labs = load_labels("data\\t10k-labels-idx1-ubyte")

    batch_size = 200
    epoch = 100

    train_dataset = GanDataset(train_imgs,train_labs)
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    discriminator = Dis_Model().to(device)
    generator = Gen_model().to(device)

    loss_fun = nn.BCEWithLogitsLoss()

    opt_d = torch.optim.Adam(discriminator.parameters(),lr=0.0001)
    opt_g = torch.optim.Adam(generator.parameters(),lr=0.0001)

    for e in range(epoch):
        for batch_imgs,batch_labels in train_dataloader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            batch_ = batch_imgs.shape[0]

            batch_real_imgs = batch_imgs.reshape(batch_,-1) # ? 为什么不用直接用 batch_size
            batch_fake_imgs = generator(batch_)

            pre1 = discriminator(batch_real_imgs)
            loss1 = loss_fun(pre1,torch.ones(batch_,1,device=device))

            pre2 = discriminator(batch_fake_imgs)
            loss2 = loss_fun(pre2,torch.zeros(batch_,1,device=device))

            loss_d = loss1 + loss2
            loss_d.backward()
            opt_d.step()

            opt_d.zero_grad()
            opt_g.zero_grad()

            # 训练生成器
            batch_fake_imgs_2 = generator(batch_)
            pre3 = discriminator(batch_fake_imgs_2)
            loss_g = loss_fun(pre3,torch.ones(batch_,1,device=device))

            loss_g.backward()
            opt_g.step()

            opt_d.zero_grad()
            opt_g.zero_grad()


        print(loss_g.item(),loss_d.item())

        imgs = generator(32)
        save_img(imgs,e)