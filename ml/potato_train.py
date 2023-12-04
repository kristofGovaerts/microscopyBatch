from __future__ import print_function, division
import os

import cv2

import torch
import torchvision
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt


from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from common.tools import bgr2gray, bgr2rgb
from ml.utils import imshow, pad

LOCATION = r'C:\Users\govaerts.kristof\programming\cuisson\0'

transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0,),
                                            (0.5,))
                   ])


# Define dataset class
class PotatoData(Dataset):
    def __init__(self, annotations_file, img_dir, im_size=(128, 128), mode='gray', transform=(0, 0.5)):
        self.img_labels = pd.read_csv(annotations_file, names=['file', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.mode=mode
        if self.mode == 'gray':
            self.size = (im_size[0], im_size[1], 1)
        elif self.mode == 'rgb':
            self.size = (im_size[0], im_size[1], 3)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        image = pad(image, to_size=list(self.size)[:2] + [3])
        if self.mode == 'gray':
            image = bgr2gray(image).astype(np.float32)/254
            self.transformation = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((self.transform[0],), (self.transform[1],))])
            image = self.transformation(image)

        elif self.mode == 'rgb':
            image = bgr2rgb(image).astype(np.float32)/254
            self.transformation = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize(
                                                          mean = 3*[self.transform[0]],
                                                          std = 3*[self.transform[1]])])
            image = self.transformation(image)

        im_name = os.path.split(img_path)[-1]
        label = self.img_labels.loc[self.img_labels.file == im_name, 'label'].iloc[0]

        return image, label


class Net(nn.Module):

    def __init__(self, input_size=(1, 128, 128)):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            # Convolve, Maxpool, Relu
            nn.Conv2d(input_size[0], 10, 5), nn.MaxPool2d(2), nn.ReLU(),
            # Convolve, Maxpool, Relu
            nn.Conv2d(10, 20, 5), nn.MaxPool2d(2), nn.ReLU()
        )

        self.features_size = self._get_collection_output_size(input_size, self.features)

        self.classifier = nn.Sequential(
            # Apply linear activation, Relu
            nn.Linear(self.features_size, 50), nn.ReLU(),
            # Apply linear activation, Relu
            nn.Linear(50, 10), nn.ReLU(),
            # Apply sigmoid in the final layer
            #             nn.Sigmoid()  # NOT GOOD FOR Multiclass classification
            nn.LogSoftmax(dim=1)
            # This is a special non-linearity that forces the output to be a probability distribution (in our case, 10 values between 0 and 1
            # that sum to 1). However, it outputs the Log of these values. This is a trick that improves the numerical stability of the code
            # and is used in combination with nn.NLLLoss() for the loss (you'll see this a bit later)
        )

    def _get_collection_output_size(self, input_size, collection):
        c = collection(Variable(torch.ones(1, *input_size)))
        # print("Size: ", c.size())
        return int(np.prod(c.size()[1:]))

    def forward(self, x):
        # Forward prop invokes the feature extraction
        x = self.features(x)
        # After feature extraction, the features are reshaped
        x = x.view(-1, self.features_size)
        # Reshaped features are passed on to the classification stage
        x = self.classifier(x)
        return x


class DenseNet121_reshaped(nn.Module):
    def __init__(self):
        super(DenseNet121_reshaped, self).__init__()
        self.densenet121 = torchvision.models.densenet121().features
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(in_features=1024, out_features=10)

    def maxpool(self, x):
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        return x

    def forward(self, x):
        x = self.densenet121(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 10 * 10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24 * 10 * 10)
        output = self.fc1(output)

        return output


def train(epoch):
    model.train()  # Sets the module in training mode.

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()  # Don't forget to clean the old gratient values

        # Feed data to the model
        output = model(data)

        # Compute loss and backpropagate
        loss = loss_fn(output, target)
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Print information evey 500
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))  # loss.data[0]))


def test():
    model.eval()  # Always use this during training
    test_loss = 0
    correct = 0
    labs = []
    preds = []
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)  # Pass the data through the model
        # print(output.shape)
        # print(target.shape)
        test_loss += F.nll_loss(output, target, size_average=False).data.item()  # [0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability, which is the class the model thinks the input is.
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()  # Test how many predicted classes match the ground truth known classes.
        pred = pred.numpy()
        pred = np.reshape(pred, (pred.shape[0]))
        target = target.numpy()
        target = np.reshape(target, (target.shape[0]))
        labs +=  list(target)
        preds += list(pred)

    test_loss /= len(test_loader.dataset)
    print("test_loss: ", test_loss)
    test_loss_list.append([test_loss])
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return preds, labs


if __name__ == '__main__':
    # initialize data + loader
    pot_data = PotatoData(r'C:\Users\govaerts.kristof\programming\cuisson\0\TCU.csv',
                          r'C:\Users\govaerts.kristof\programming\cuisson\0',
                          mode='rgb',
                          transform=(0.5, 0.5))

    train_size = int(len(pot_data) * 0.7)
    test_size = int(len(pot_data) * 0.15)
    val_size = len(pot_data) - train_size - test_size
    pot_train, pot_test, pot_val = random_split(pot_data, [train_size, test_size, val_size])

    # num_workers = 0 since we are not working with multiple cores
    train_loader = torch.utils.data.DataLoader(dataset=pot_train, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=pot_test, batch_size=128, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=pot_val, batch_size=128, shuffle=True, num_workers=0)

    # get some random training images
    dataiter = iter(train_loader)
    sample = next(dataiter)
    images = sample[0]
    labels = sample[1]

    imshow(torchvision.utils.make_grid(images))
    plt.show()

    # define neural network architecture

    img_size = images[0].shape
    model = Net(img_size)
    model = torchvision.models.densenet121()

    #print(model)

    params = list(model.parameters())
    print(len(params))
    print(params[-1].shape)  # Shape of the Final layer (should be 10)
    print(params[-1])

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    # optimizer = optim.RMSprop(params, lr=0.001)
    # optimizer = optim.Adadelta(params, lr=0.001)
    optimizer = optim.Adam(params, lr=0.0001, betas=(0.9, 0.999))

    loss_fn = nn.NLLLoss()  # This loss matches the output nn.LogSoftmax we used earlier

    test_loss_list = []

    max_epochs = 20
    for epoch in range(1, max_epochs + 1):
        train(epoch)
        test_preds, test_labs = test()

    # Let's visualize some random test images
    dataiter = iter(test_loader)
    test_images, labels = next(dataiter)

    print(test_images.shape)

    # show images
    plt.figure(figsize=(10, 10))

    test_image_set = test_images[0:5]
    imshow(torchvision.utils.make_grid(test_image_set))

    model.eval()
    outputs = model(Variable(test_image_set))
    _, predicted = torch.max(outputs.data, 1)

    classes = pot_data.img_labels.label.unique()
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(5)))
