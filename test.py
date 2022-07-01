import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import ResNet18

use_cuda = False    # 控制是否需要使用GPU模式
transform_test = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))     # 归一化
                             ])
# 测试集
testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False)

model_path = './checkpoint/epoch=5_iter=200_acc=0.921.pth'  # 预训练权重的位置
net = ResNet18()    # 实例化网络对象

if use_cuda:    # 如果使用了GPU模式，需要把定义的网络放到GPU上
    net = net.cuda()
    checkpoint = torch.load(model_path)
else:   # 由于我是在GPU上训练的，因此使用CPU加载时需要把与与训练权重转换为cpu版本
    checkpoint = torch.load(model_path, map_location='cpu')
# 网络加载预训练权重
net.load_state_dict(checkpoint)

correct = 0  # 预测正确的图片数
total = 0  # 总共的图片数

for data in testloader:
    images, labels = data

    # 同理，如果使用GPU，需要把待处理的图片和标签也放在GPU上
    if use_cuda:
        images = images.cuda()
        labels = labels.cuda()

    outputs = net(images)   # 输出网络的预测结果
    predicted = torch.argmax(outputs, 1)    # 找到预测结果中概率最大的值对应的位置，即最终分类
    total += labels.size(0)
    correct += (predicted == labels).sum()  # 记录预测的分类与定义label相等的个数，即预测正确的个数
acc = torch.tensor(correct, dtype=torch.float32) / total    # 计算百分比
print('测试集中的准确率为: %.3f ' % (acc))
