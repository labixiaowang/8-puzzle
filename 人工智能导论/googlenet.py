import time
import torch
import numpy as np
from torch import nn, optim
import torchvision
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

###################fashion mnist数据集加载######################
def load_data_fashion_mnist(batch_size, resize=None, root='Datasets'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))  # 调整图形大小
    trans.append(torchvision.transforms.ToTensor())
    # 图像增强
    transform = torchvision.transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_iter, test_iter


#################################################################
batch_size = 32
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)


# 图像展示
from matplotlib import pyplot as plt
from IPython import display

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels] # 获取第几类

# def show_fashion_mnist(images, labels):
#     """Use svg format to display plot in jupyter"""
#     display.set_matplotlib_formats('svg')
#     # 这里的_表示我们忽略（不使用）的变量
#     _, figs = plt.subplots(1, len(images), figsize=(12, 12))
#     for f, img, lbl in zip(figs, images, labels):
#         f.imshow(img.view((96, 96)).numpy())
#         f.set_title(lbl)
#         f.axes.get_xaxis().set_visible(False)
#         f.axes.get_yaxis().set_visible(False)

# 读取训练数据集中第一个batch的数据
# train_data = iter(train_iter)
# images, labels = next(train_data)
# # 观察训练数据集中前10个样本的图像内容和文本标签
# labels = get_fashion_mnist_labels(labels)
# show_fashion_mnist(images[:10], labels[:10])
# plt.show()

class Residual(nn.Module):
    # 可以设定输出通道数、是否使用额外的1x1卷积层来修改通道数以及卷积层的步幅。
    def __init__(self, in_c,c1, c2, c3, c4, use_1x1conv=False, stride=1):
        '''
        :param in_c: 输入通道数
        :param out_c:输出通道数
        :param c1: 线路1的卷积层输出通道数
        :param c2: 线路2的卷积层输出通道数
        :param c3: 线路3的卷积层输出通道数
        :param c4: 线路4的卷积层输出通道数
        :param use_1x1conv:
        '''
        super(Residual, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1, stride=stride)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1, stride=stride)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1, stride=stride)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1, stride=stride)

        # if use_1x1conv:
        #     self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride)
        # else:
        #     self.conv1 = None
        # self.bn = nn.BatchNorm2d(out_c)

    def forward(self, X):
        p1 = F.relu(self.p1_1(X))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(X))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(X))))
        p4 = F.relu(self.p4_2(self.p4_1(X)))
        Y = torch.cat((p1, p2, p3, p4), dim=1)# 沿着通道维度拼接
        return Y


def resnet_block(in_c, out_c, c1, c2, c3, c4, num_residuals, first_block=False):
    if first_block:
        assert in_c == out_c # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_c, out_c, c1, c2, c3, c4, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_c, out_c, c1, c2, c3, c4))
    return nn.Sequential(*blk)  # *解包


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
    # x(batch_size, channels, height, width)


class FlattenLayer(torch.nn.Module):  #展平操作
    def forward(self, x):
        return x.view(x.shape[0], -1)

# torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。
# net = nn.Sequential(
#     	# 添加第一个卷积层，调用了nn里面的Conv2d()
#         nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#         # 进行数据的归一化处理
#     	nn.BatchNorm2d(32),
#     	# 修正线性单元，是一种人工神经网络中常用的激活函数
#         nn.ReLU(),
#     	# 再进行最大池化处理
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# # 依次添加resnet_block模块
# net.add_module("resnet_block1", resnet_block(32, 32, 8, (4, 8), (4, 8), 8, 2, first_block=True))
# net.add_module("resnet_block2", resnet_block(32, 80, 16, (16, 32), (8, 16), 16, 2))
# net.add_module("resnet_block3", resnet_block(80, 192, 32, (32, 64), (32, 64), 32, 2))
# net.add_module("resnet_block4", resnet_block(192, 320, 64, (64, 128), (32, 64), 64, 2))
# # 添加GlobalAvgPool2d模块
# net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 256, 1, 1)
# # 添加FlattenLayer模块，再接一个全连接层
# net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(320, 10)))
# # 模型定义-ResNet
# print(net)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b3 = nn.Sequential(Residual(192, 64, (96, 128), (16, 32), 32),
                   Residual(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(Residual(480, 192, (96, 208), (16, 48), 64),
                   Residual(512, 160, (112, 224), (24, 64), 64),
                   Residual(512, 128, (128, 256), (24, 64), 64),
                   Residual(512, 112, (144, 288), (32, 64), 64),
                   Residual(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b5 = nn.Sequential(Residual(832, 256, (160, 320), (32, 128), 128),
                   Residual(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

def evaluate_accuracy(data_iter, net, device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        # If device is the GPU, copy the data to the GPU.
        X, y = X.to(device), y.to(device)
        net.eval()
        with torch.no_grad():
            y = y.long()
            # [[0.2 ,0.4 ,0.5 ,0.6 ,0.8] ,[ 0.1,0.2 ,0.4 ,0.3 ,0.1]] => [ 4 , 2 ]
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))
            n += y.shape[0]
    return acc_sum.item() / n

def train_ch(net, train_iter, test_iter, criterion, num_epochs, device, lr=None):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 优化函数
    best_test_acc = 0
    for epoch in range(num_epochs):
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)  # 损失
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        n, start = 0, time.time()
        for X, y in train_iter:
            net.train()
            optimizer.zero_grad()  # 清空梯度
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                y = y.long()  # 将张量转化成long类型
                train_l_sum += loss.float()
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == y))).float()
                n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net, device)  # 测试验证集
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))
        if test_acc > best_test_acc:
            # print('find best! save at model/best.pth')
            best_test_acc = test_acc
            torch.save(net.state_dict(), 'model/best3.pth')

# 超参数设置
lr, num_epochs = 0.1, 7
criterion = nn.CrossEntropyLoss()   #交叉熵描述了两个概率分布之间的距离，交叉熵越小说明两者之间越接近
train_ch(net, train_iter, test_iter, criterion, num_epochs, device, lr)