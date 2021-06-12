from model import *
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#net = Net()
net = torchvision.models.vgg16(pretrained=False)
#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, 2)
net.classifier[6] = nn.Linear(4096, 2)
net = net.to(device)
PATH = './weight/weight_gpu_vgg16_resize_weight.pth'
net.load_state_dict(torch.load(PATH))

trans = transforms.Compose([transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_set = torchvision.datasets.ImageFolder(root='./binary/test',
                                            transform=trans)
test_loader = DataLoader(test_set, batch_size=1,
                          shuffle=True, num_workers=2)


def main():
    correct = 0
    total = 0
    cls_total = [0, 0]
    cls_correct = [0, 0]
    cls_predict = [0, 0]
    with torch.no_grad():
        start_time = time.time()
        for data in test_loader:
            images, labels = data[0].to(device=device), data[1].to(device=device)
            outputs = net(images)
            _, prediction = torch.max(outputs, 1)
            cls_total[labels] += 1
            cls_predict[prediction] += 1
            if prediction == labels:
                correct += 1
                cls_correct[labels] += 1
            total += 1
        end_time = time.time()
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    print('recall')
    for i in range(2):
        print('Accuracy of the network on the test images: %d %%' % (
                100 * cls_correct[i] / cls_total[i]))
    print('precision')
    for i in range(2):
        print('Accuracy of the network on the test images: %d %%' % (
                100 * cls_correct[i] / cls_predict[i]))
    print(end_time - start_time)


if __name__ == '__main__':
    main()