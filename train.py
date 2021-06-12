
from model import *
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim


def main():
    trans = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.ImageFolder(root='./binary/train',
                                                 transform=trans)
    train_loader = DataLoader(train_set, batch_size=4,
                              shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    #net = Net()
    net = torchvision.models.vgg16(pretrained=False)
    #num_ftrs = net.fc.in_features
    #net.fc = nn.Linear(num_ftrs, 2)
    net.classifier[6] = nn.Linear(4096, 2)
    net = net.to(device)
    print(device)

    PATH = './weight_gpu_vgg16_resize_weight3565.pth'

    weight = torch.tensor([0.35, 0.65]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-4)
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device=device), data[1].to(device=device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    main()