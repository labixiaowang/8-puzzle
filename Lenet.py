import torch.nn as nn
import torch
from torchvision import datasets,transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 2##########加载MNIST数据集,这个就是最终的模型，接下来就
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

model=LeNet5()
print(model)

#%%

# 定义模型、损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
i=0
# 3################训练模型
for epoch in range(5):
    for i, (images, labels) in enumerate(train_loader):
        #print(i)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, 10, i + 1, len(train_loader),
                                                                     loss.item()))
# 4######################测试模型
model.eval()
i=0
incorrect_images = []
incorrect_labels = []
correct_labels = []
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        #print(i)
        #i=i+1
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # incorrect_indices = (predicted != labels).nonzero().view(-1)
        # incorrect_images.extend(images[incorrect_indices].cpu())
        # incorrect_labels.extend(predicted[incorrect_indices].cpu())
        # correct_labels.extend(labels[incorrect_indices].cpu())
        # # 显示前几个错误预测的图像
        # num_to_show = 10
        # grid_tensor = make_grid(incorrect_images[:num_to_show], nrow=5)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(transforms.ToPILImage()(grid_tensor))
        # plt.axis('off')
        # plt.title('Misclassified Images')


        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
