import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from project1_model import project1_model


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
  # torch.numel() returns number of elements in a tensor

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = train_loss/len(trainloader)
    
    train_loss_history.append(train_loss)
    print('Training Loss: %.4f | Acc: %.4f' % (train_loss, 100.*correct/total))


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss/len(testloader)
    test_acc = 100.*correct/total
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)
    print('Testing Loss: %.4f | Acc: %.4f' % (test_loss, test_acc))


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # data load
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')


    net = project1_model()
    print(count_parameters(net))
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_loss_history = []
    test_loss_history = []
    test_acc_history = []

    num_epoch = 80
    for epoch in range(num_epoch):
        train(epoch)
        test(epoch)
        scheduler.step()

    print("train: ", train_loss_history)
    print("test: ", test_loss_history)

    # plot
    plt.plot(range(num_epoch),train_loss_history,'-',linewidth=3,label='Train error')
    plt.plot(range(num_epoch),test_loss_history,'-',linewidth=3,label='Test error')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.savefig("./loss_epoch.png")

    plt.plot(range(num_epoch), test_acc_history, '-', linewidth=3, label='test accuracy')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig("./acc_epoch.png")

    # save model
    model_path = "./project1_model.pt"
    torch.save(net.module.state_dict(), model_path)