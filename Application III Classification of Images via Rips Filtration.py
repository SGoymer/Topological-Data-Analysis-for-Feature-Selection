# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:24:08 2021

@author: Sam
"""

#%%

import numpy as np
import time

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from ripser import ripser, lower_star_img

import persim
from persim import plot_diagrams, PersistenceImager

from random import seed
from random import random

import torch
import torchvision
from torch.utils.data import Dataset

#%%
#rbg2gray function. Converts XxYx3 coloured image array into XxY grayscale image array.
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#datasetToArray function. Converts torchvision dataset (dataset) to 28x28 ndarrays.
#size: int. If specified, returns only the first (size) elements.
def datasetToArray(dataset, size = None):
    X=np.zeros((len(dataset),28,28))
    y=np.zeros(len(dataset))
    for i, datapoint in enumerate(dataset):
        X[i] = np.array(datapoint[0]).astype(np.float)
        y[i] = datapoint[1]
    X = X/255.
    X = -X + 1
    if size == None:
        return X, y
    else:
        return X[:size], y[:size]
    
#Thresh function. Randomly adds (pointsPerSquare) points to each pixel in diagram with grayscale value darker than (threshold).
#For more details, see the paper (Application III)
def Thresh(images, threshold, pointsPerSquare = 5, random_state = 1, display = 0, pointChance = 1):
    pointsArrays = []
    seed(random_state)
    for imgno, img in enumerate(images):
        pointArray = np.ndarray(shape=(0,2))
        for row, rowdata in enumerate(img):
            for column, datapoint in enumerate(rowdata):
                if datapoint < threshold:
                    for n in range(pointsPerSquare):
                        if random() < pointChance:
                            pointArray = np.vstack([pointArray, [column+random(), row + random()]])
        pointsArrays.append(pointArray)
        if display != 0:
            if display == -1:
                display = len(images)
            plt.imshow(img, cmap = 'gray')
            plt.scatter(pointArray[:,0], pointArray[:,1])
            plt.show()
    return pointsArrays
#%%
#Load image datasets. 
train_size = 60000
test_size = 10000
dataset_train = torchvision.datasets.FashionMNIST('/files/', train=True, download=True)
dataset_test = torchvision.datasets.FashionMNIST('/files/', train=False, download=True)
x_train_transformed, y_train = datasetToArray(dataset_train, size = train_size)
x_test_transformed, y_test = datasetToArray(dataset_test, size = test_size)
data_subset = [5,7,8]
train_size = 1800
test_size = 300
x_train_transformed = [img for i, img in enumerate(x_train_transformed) if y_train[i] in data_subset][:train_size]
y_train = [y for y in y_train if y in data_subset][:train_size]
x_test_transformed = [img for i, img in enumerate(x_test_transformed) if y_test[i] in data_subset][:test_size]
y_test = [y for y in y_test if y in data_subset][:test_size]
train_pointsArrays = Thresh(x_train_transformed, pointsPerSquare=3, threshold = 0.8, display = 0)
test_pointsArrays = Thresh(x_test_transformed, pointsPerSquare=3, threshold = 0.8, display = 0)

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
#Parameters:
    #train_data, test_data: The point cloud datasets. If ripsVersion = True, list of XxY grayscale image data. If ripsVersion = False, 2xN array of 2d point cloud data.
    #spread: Standard deviation of Gaussian function in the PI trnasformation. Unlikely to affect accuracy.
    #pixels: Output size of persistence image.
    #display: number of PDs/PIs to display (displays first <display> images).
    #yValue: if not equal to -1, displays only PIs with this target value.
    #persThreshold: None or float. If float, removes all classes from the PD with class persistence below the threshold.
    #pixel_size: Images produced have fixed dimension. Smaller pixel size => higher resolution.
    #weight: Weight parameter for PI transformation. Larger weight increases significance of large-persistence classes.
#Returns:
    #dgmlist: list of PDs from both datasets
    #two output[:]s: list of train and test PIs
def calculatePIs(train_data, test_data, spread = 1, pixels = [28,28], display = 0, yValue = -1, persThreshold = None, ripsVersion = False, pixel_size=0.1, weight=1.2):
    output = []
    dgmlist = []
    pim = persim.PersImage(spread=spread, pixels=pixels, verbose=False)
    pimgr = PersistenceImager(pixel_size = pixel_size, weight_params={'n': weight}, kernel_params={'sigma': [[spread, 0.0], [0.0, spread]]})
    for i, img in enumerate(train_data + test_data):
        if i % 10 == 0:
            print('creating image number: ' + str(i))
            timer = time.perf_counter()
        if ripsVersion == False:
            dgm = lower_star_img(img)
            dgm = dgm[np.isinf(dgm[:,1]) == False, :]
            persimg = pim.transform(dgm)
            output.append(persimg)
            if i < display:
                if yValue == -1 or y_train[i] == yValue:
                    plot_diagrams(dgm)
                    plt.title('Persistence Diagram index: ' + str(i) + ' Target Value: ' + str(y_train[i]))
                    plt.show()
                    pim.show(persimg)
                    plt.title('Persistence Image index: ' + str(i) + ' Target Value: ' + str(y_train[i]))
                    plt.show()
        if ripsVersion == True:
            dgm = ripser(img)['dgms'][1]
            if persThreshold != None:
                cut_dgm = [pair for pair in dgm if pair[1] - pair[0] > persThreshold]
                if cut_dgm == []:
                    cut_dgm = [dgm[0]]
                dgm = cut_dgm
            #print('number of pairs: '+ str(len(dgm)))
            dgmlist.append(dgm)
        if i % 10 == 0:
            print('iteration time in seconds: ' + str(time.perf_counter() - timer))
    if ripsVersion == True:
        output = pimgr.fit_transform(dgmlist, skew=True)
        if display != 0 and yValue == -1:
            for i, PI in enumerate(output[:display]):
                plot_diagrams(dgmlist[i])
                plt.title('Persistence Diagram index: ' + str(i) + ' Target Value: ' + str(y_train[i]))
                plt.show()
                pim.show(PI)
                plt.title('Persistence Image index: ' + str(i) + ' Target Value: ' + str(y_train[i]))
                plt.show()
    return(dgmlist, output[:len(train_data)], output[len(train_data):])

#Combine function. Reduces number of rows in PI by adding (numRowsximagewidth) blocks of image together modularly.
#Parameters:
    #adapted_PIs: the PIs to convert.
    #numRows: number of rows in ouptut image.
#Returns:
    #output_list: list of reduced PIs.
def Combine(adapted_PIs, numRows = 10):
    output_list = []
    for PI in adapted_PIs:
        output = torch.zeros([1, numRows, PI.shape[2]])
        PI_row_blocks = int((PI.shape[1] - (PI.shape[1] % numRows))/numRows)
        #spare_rows = PI.shape[1] % 10
        for row_index in range(PI_row_blocks):
            output = output + PI[:, numRows*row_index : numRows*(row_index + 1), :]
        output_list.append(output)
    return output_list

#%%
    
persistence_diagrams, x_train_PIs, x_test_PIs = calculatePIs(train_pointsArrays, test_pointsArrays, display = 10, spread = 0.1, pixel_size = 0.2, weight = 1.2, ripsVersion = True)
print('PIs done!')
train_mean = np.mean(x_train_PIs)
train_std = np.std(x_train_PIs)
#normalise data, convert to tensor and add dimension of size 1
x_train_PIs_adapted = list(map(lambda x: torch.tensor(((x - train_mean)/train_std).copy()).unsqueeze(0), x_train_PIs))
#convert to numpy array
x_train_PIs_adapted2 = list(map(lambda x: x.squeeze().numpy(), Combine(x_train_PIs_adapted, numRows = 1)))


test_mean = np.mean(x_test_PIs)
test_std = np.std(x_test_PIs)
x_test_PIs_adapted = list(map(lambda x: torch.tensor(((x - test_mean)/test_std).copy()).unsqueeze(0), x_test_PIs))
x_test_PIs_adapted2 = list(map(lambda x: x.squeeze().numpy(), Combine(x_test_PIs_adapted, numRows = 1)))

#%%
#Augment function. Turns list of tensors (data) and list of target values (targets) into list of lists with elements (data, target value).

def augment(data, targets):
     output = []
     for i, datapoint in enumerate(data):
         output.append([datapoint.float(), int(targets[i])])
     return output
def augmentn(data, targets):
    output = []
    for i, datapoint in enumerate(data):
        output.append([datapoint, int(targets[i])])
    return output
#%%
#create Datasets
train_data = augmentn(x_train_PIs_adapted2, y_train)
test_data = augmentn(x_test_PIs_adapted2, y_test)
x_train_dataset = PIsDataset(train_data)
x_test_dataset = PIsDataset(test_data)

#%%
#Train SVM on train datset and predict values for test dataset
from sklearn import svm
clf = svm.SVC()
clf.fit(x_train_PIs_adapted2, y_train)
y_pred = clf.predict(x_test_PIs_adapted2)
print('SVM correct predictions: ' + str(sum(y_test == y_pred)) + '/' + str(len(y_pred)) + \
      '. Accuracy: ' + str(sum(y_test == y_pred)/len(y_test)))
print('\nConfusion Matrix:\n')
print(confusion_matrix(y_test, y_pred))

#%%
#Find optimal parameters for SVM
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[5, 10, 15, 20]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(x_train_PIs_adapted2, y_train)


for key, value in clf.cv_results_.items():
    print(str(key) + "s: " + str(value))
    
import pandas as pd
pd.concat([pd.DataFrame(clf.cv_results_["params"]),pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Accuracy"]), pd.DataFrame(clf.cv_results_["rank_test_score"], columns=["Rank"])],axis=1)
