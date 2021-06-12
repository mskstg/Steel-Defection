from model import *
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#net = Net()
net = torchvision.models.resnet18(pretrained=False)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
net = net.to(device)
PATH = './weight/weight_gpu_resnet18_2.pth'
net.load_state_dict(torch.load(PATH))

trans = transforms.Compose([transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_set = torchvision.datasets.ImageFolder(root='./binary/val',
                                            transform=trans)
test_loader = DataLoader(test_set, batch_size=1,
                          shuffle=True, num_workers=2)


def main():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device=device), data[1].to(device=device)
            outputs = net(images)
            if outputs.data[0][0] >= outputs.data[0][1]:
                predicted = 0
            else:
                predicted = 1

            total += 1
            if predicted == labels:
                correct += 1

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    main()