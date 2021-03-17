# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:26:59 2021

@author: Sam
"""

#%%

import numpy as np
import time

import matplotlib.image as mpimg
import glob
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from ripser import ripser, lower_star_img

import persim
from persim import plot_diagrams

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

#%%
#rbg2gray function. Converts XxYx3 coloured image array into XxY grayscale image array.
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
#%%
#Load image datasets. Be sure to add the cell_images folder to your working directory.
uninfected_images = list(map(mpimg.imread, glob.glob('cell_images/Uninfected/*.png')))
parasitised_images = list(map(mpimg.imread, glob.glob('cell_images/Parasitized/*.png')))
uninfected_gimages = list(map(rgb2gray, uninfected_images))
parasitised_gimages = list(map(rgb2gray, parasitised_images))
x = uninfected_gimages + parasitised_gimages
y = [0]*len(uninfected_images) + [1]*len(parasitised_images)
try:
    x_train_transformed, x_test_transformed, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, shuffle = True)
except ValueError:
    print("No images were found when attempting to create train-test split. Did you add the cell_images folder to your working directory?")

#%%
class PIsDataset(Dataset):
    """Persistence Images dataset for use with sklearn."""

    def __init__(self, data_source, transform = None):
        self.PIs = data_source
        self.transform = transform

    def __len__(self):
        return len(self.PIs)

    def __getitem__(self, idx):
        sample = self.PIs[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample

#calculatePIs function. Produces persistence images from image dataset or point cloud (ripsVersion).
#note: Uses deprecated PersImage class. See Application III for use of new PersistenceImager class.
#Parameters:
    #data: The image/point cloud dataset. If ripsVersion = True, list of XxY grayscale image data. If ripsVersion = False, 2xN array of 2d point cloud data.
    #spread: Standard deviation of Gaussian function in the PI trnasformation. Unlikely to affect accuracy.
    #pixels: Output size of persistence image.
    #display: number of PDs/PIs to display (displays first <display> images).
    #yValue: if not equal to -1, displays only PIs with this target value.
#Returns:
    #output: list of ndarrays (PIs) with shape = shape(pixels).
def calculatePIs(data, spread = 1, pixels = [28,28], display = 0, yValue = -1, ripsVersion = False):
    functimer = time.perf_counter()
    output = []
    pim = persim.PersImage(spread=spread, pixels=pixels, verbose=False)
    for i, img in enumerate(data):
        if i % 10 == 0:
            print('Creating persistence image number: ' + str(i))
            timer = time.perf_counter()
        if ripsVersion == False:
            dgm = lower_star_img(img)
            dgm2 = dgm[np.isinf(dgm[:,1]) == False, :]
            persimg = pim.transform(dgm2)
            output.append(persimg)
            if i < display:
                if yValue == -1 or y_train[i] == yValue:
                    plot_diagrams(dgm2)
                    plt.title('Persistence Diagram index: ' + str(i) + ' Target Value: ' + str(y_train[i]))
                    plt.show()
                    pim.show(persimg)
                    plt.title('Persistence Image index: ' + str(i) + ' Target Value: ' + str(y_train[i]))
                    plt.show()
        if ripsVersion == True:
            dgm = ripser(img)['dgms']
            #plot_diagrams(dgm)
            #plt.title('img no. ' + str(i) + ', true value: ' + str(y_train[i]))
            #plt.show()
            dgm_0 = dgm[0]
            dgm_1 = dgm[1]
            dgm_0_2 = dgm_0[np.isinf(dgm_0[:,1]) == False, :]
            dgm_1_2 = dgm_1[np.isinf(dgm_1[:,1]) == False, :]
            persimg_0 = pim.transform(dgm_0_2)
            persimg_1 = pim.transform(dgm_1_2)
            persimg = np.hstack((persimg_0, (persimg_0.max()/persimg_1.max())*persimg_1))
            output.append(persimg)
        if i % 100 == 0:
            print('Iteration time in seconds: ' + str(time.perf_counter() - timer))
    print('function time in seconds: ' + str(time.perf_counter() - functimer))
    return(output)

#%%
#create test dataset PIs
x_train_PIs = calculatePIs(x_train_transformed, spread = 0.1, display = 10, yValue = -1)
print('train PIs done!')
train_mean = np.mean(x_train_PIs)
train_std = np.std(x_train_PIs)
#normalise data, convert to tensor and add dimension of size 1
x_train_PIs_adapted = list(map(lambda x: torch.tensor(((x - train_mean)/train_std).copy()).unsqueeze(0), x_train_PIs))

x_test_PIs = calculatePIs(x_test_transformed, spread = 0.1, display = 10, yValue = -1)
print('test PIs done!')
test_mean = np.mean(x_test_PIs)
test_std = np.std(x_test_PIs)
x_test_PIs_adapted = list(map(lambda x: torch.tensor(((x - test_mean)/test_std).copy()).unsqueeze(0), x_test_PIs))

#%%
#Augment function. Connverts list of tensors (data) and list of target values (targets) into list of lists with elements (data, target value).
def augment(data, targets):
     output = []
     for i, datapoint in enumerate(data):
         output.append([datapoint.float(), int(targets[i])])
     return output

#augment PIs and convert to sklearn Dataset
train_data = augment(x_train_PIs_adapted, y_train)
test_data = augment(x_test_PIs_adapted, y_test)
x_train_dataset = PIsDataset(train_data)
x_test_dataset = PIsDataset(test_data)

#%%
#NN hyperparameters
n_epochs = 20
batch_size_train = 64
batch_size_test = 400
learning_rate = 0.1
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

#%%
#load datasets
train_loader = torch.utils.data.DataLoader(x_train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(x_test_dataset, batch_size=batch_size_test, shuffle=True)

#%%

#Build NN

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, -1)
    
#%%

#Initialize network and optimiser
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

#%%

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
test_accs = []

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'results.pth')
      torch.save(optimizer.state_dict(), 'optimiser.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  test_acc = 100. * correct / len(test_loader.dataset)
  test_accs.append(test_acc)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    test_acc))
  print('\nConfusion Matrix:\n')
  print(confusion_matrix(target, pred))
#%%
  
#train and test the network
  
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

#%%
#plot results
fig, ax1 = plt.subplots()
ax1.plot(train_counter, train_losses, color='blue')
ax1.scatter(test_counter[1:], test_losses, color='red')
ax1.legend(['Train Loss', 'Test Loss'], loc='right')
ax1.set_xlabel('number of training examples seen')
ax1.set_ylabel('negative log likelihood loss')
ax2 = ax1.twinx()
ax2.plot(test_counter[1:], test_accs, color='green')
ax2.set_ylabel('% Accuracy')
ax2.set_yticks(np.arange(50., 100., 5.))
ax2.legend(['Test Accuracy'], loc=[0.703,0.37])
fig.tight_layout()
plt.show()
