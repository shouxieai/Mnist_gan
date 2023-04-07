import numpy as np
import struct
import matplotlib.pyplot as plt
import torch
import os
import random
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision
import torch.nn.functional as func
from tqdm import tqdm


def load_labels(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)


def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, 28,28)


class GDataset(Dataset):
    def __init__(self,data,label):
        self.all_data = data
        self.all_label = label

    def __getitem__(self, index):
        data = self.all_data[index]
        label = self.all_label[index]

        data = data.astype(np.float32)
        label = label.astype(np.int32)

        return data,label

    def __len__(self):
        return len(self.all_label)


class Dis_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784,1024)
        self.lrelu = nn.LeakyReLU()
        self.linear2 = nn.Linear(1024,1)
    def forward(self,x):
        x = self.linear1(x)
        x = self.lrelu(x)
        x = self.linear2(x)

        return x

class Gen_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num1 = 512
        self.num2 = 1024


        self.linear1 = nn.Linear(self.num1,self.num2)
        self.lrelu = nn.LeakyReLU()
        self.linear2 = nn.Linear(self.num2,784)

    def forward(self,size):
        x = torch.randn(size=(size[0],self.num1),device=self.linear2.weight.device)

        x = self.linear1(x)
        x = self.lrelu(x)
        x = self.linear2(x)

        return x

def save_img(img,e):
    """打印生成器产生的图片"""
    # torchvision.utils.make_grid用来连接一组图， img为一个tensor（batch, channel, height, weight）
    # .detach()消除梯度

    img = img[:32]

    out = img.view(img.shape[0], 28, 28).unsqueeze(1)  # 添加channel
    img = out.clamp(0, 1)

    im = torchvision.utils.make_grid(img, nrow=8).detach().cpu().numpy()
    # print(np.shape(im))
    plt.title(f"Epoch on {epoch + 1}")
    plt.imshow(im.transpose(1, 2, 0))  # 调整图形标签， plt的图片格式为(height     , weight, channel)
    plt.savefig(f'./result/{e + 1}.jpg')



if __name__ == "__main__":
    train_datas = load_images("data\\train-images-idx3-ubyte") / 255
    train_label = load_labels("data\\train-labels-idx1-ubyte")

    test_datas = load_images("data\\t10k-images-idx3-ubyte") / 255
    test_label = load_labels("data\\t10k-labels-idx1-ubyte")

    train_dataset = GDataset(train_datas, train_label)
    test_dataset = GDataset(test_datas, test_label)

    epoch = 100
    batch_size = 100

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    discriminator = Dis_Model().to(device)
    generator = Gen_Model().to(device)
    loss_fun = nn.BCEWithLogitsLoss().to(device)


    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=0.0001)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=0.0001)

    for e in range(epoch):
        for batch_datas, batch_label in tqdm(train_dataloader):
            batch_datas = batch_datas.to(device)

            real_img = batch_datas.reshape(-1,784)

            fake_img = generator.forward(size=(batch_size,784))

            pre1 = discriminator.forward(fake_img)
            pre2 = discriminator.forward(real_img)

            lossd1 = loss_fun(pre1,torch.zeros(batch_size,1,device=device))
            lossd2 = loss_fun(pre2,torch.ones(batch_size,1,device=device))

            loss_d = lossd1 + lossd2


            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            opt_d.zero_grad()

            fake_img = generator.forward(size=(batch_size, 784))
            pre3 = discriminator(fake_img)

            loss_g = loss_fun(pre3,torch.ones(batch_size,1,device=device))

            loss_g.backward()
            opt_g.step()

            opt_g.zero_grad()
            opt_d.zero_grad()

        img = generator.forward(size=(batch_size, 784))
        save_img(img,e)
