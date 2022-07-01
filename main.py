import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from torch import optim
from model import AlexNet   # AlexNet 网络
from model import ResNet18  # resnet18网络

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 定义对数据的预处理
transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),     # 归一化
                             ])
# 训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True)

transform_test = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))     # 归一化
                             ])
# 测试集
testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False)

'''
验证代码与test代码基本一致
'''
def validation():
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            outputs = net(images)
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    acc = torch.tensor(correct, dtype=torch.float32) / total
    # print('测试集中的准确率为: %.3f ' % (acc))
    return acc

def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = 0.0001 * (0.1 ** (epoch // 10))
    return lr


def train(epoches):
    running_loss = 0.0
    for epoch in range(epoches):
        with torch.enable_grad():
            for i, data in enumerate(trainloader):
                # 输入数据
                inputs, labels = data
                if use_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # 梯度清零
                optimizer.zero_grad()
                # forward + backward
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                # 更新参数
                optimizer.step()
                # 打印log信息
                # loss 是一个scalar,需要使用loss.item()来获取数值
                running_loss = 0.0
                if i % 100 == 99:  # 每100个batch打印一下训练状态
                    running_loss += loss.item()
                    # 每100个batch存储一次模型参数, 并进行一次验证
                    acc = validation()
                    print('=>epoch=%d_iteration=%d_loss=%.3f_acc=%.3f' % (epoch + 1, i + 1, running_loss / 100, acc))
                    save_path = os.path.join('./checkpoint', 'epoch=' + str(epoch+1) + '_iter='+str(i+1)+'_acc='+str(round(acc.item(), 3)) + '.pth')
                    torch.save(net.state_dict(), save_path)  # 存储网络模型的参数

    print('Finished Training')


if __name__ == '__main__':
    use_cuda = False

    '''
    使用自定义的网络，这里是参考的resnet18网络结构
    '''
    model_path = './checkpoint/epoch=5_iter=200_acc=0.921.pth' # 预训练模型
    net = ResNet18()    # AlexNet网络的用法与ResNet一致
    criterion = nn.CrossEntropyLoss()   # 交叉熵损失函数
    if use_cuda:
        net = net.cuda()
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint)     # 在预训练模型的基础上再次开始训练
        criterion = criterion.cuda()
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint) # 在预训练模型的基础上再次开始训练

    optimizer = optim.Adam(net.parameters(), lr=0.0001)     # adam优化器

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')     # cifar-10的10种分类
    train(50)   # 训练50个epoch
