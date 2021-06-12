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
net = torchvision.models.vgg19(pretrained=False)
#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, 2)
net.classifier[6] = nn.Linear(4096, 5)
net = net.to(device)
PATH = './weight_gpu_vgg19_resize_multi_weight.pth'
net.load_state_dict(torch.load(PATH))

trans = transforms.Compose([transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_set = torchvision.datasets.ImageFolder(root='./multi/train',
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
            _, prediction = torch.max(outputs, 1)
            if prediction == labels:
                correct += 1
            total += 1

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    main()