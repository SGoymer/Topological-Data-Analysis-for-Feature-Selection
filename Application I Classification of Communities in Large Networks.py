# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:36:16 2021

@author: Sam
"""

#%%
import networkx as nx
import community as pl
import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt

import time
import persim
from persim import plot_diagrams

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
#%%

#BestPartition function. Loads network from SNAP .txt file into nx.Graph(), summarises network data and creates a partition dictionary
#Parameters: graphName: name of .txt file containing edge list
#Returns:
    #graph: NetworkX graph from .txt data
    #partition: dictionary of {node index: community index} specifying which community/subnetwork each node belongs to
#note: this function is very slow and will likely take around an hour to partition both graphs.
def BestPartition(graphName):
    print('Details for graph ' + graphName + ':')
    graph = nx.read_edgelist(graphName, nodetype = int)
    print(nx.info(graph))
    print('Partitioning...')
    partition = pl.best_partition(graph, randomize = False)
    print('done')
    return graph, partition

#Calculate the number of clusters in a partition dictionary
#if reutrnFreqs is True, also creates and returns dictionary of {partitionno:frequency}
def numBPClusters(partition, returnFreqs = False, silent = False):
    max=0
    partitionFreqs = {}
    for key, value in partition.items():
        if value > max:
            max = value
        if returnFreqs == True:
            if value in partitionFreqs:
                partitionFreqs[value] = partitionFreqs[value] + 1
            else:
                partitionFreqs[value] = 1
    if silent == False:
        print('Number of clusters in best_partition: ' + str(max + 1))
    if returnFreqs == False:
        return max + 1
    if returnFreqs == True:
        return max + 1, partitionFreqs

#%%

epsilon = 0.00000001
tol = 0.1
#Applies lower star filtration to a networkx graph G. Returns gudhi Simplex Tree data structure.
def toSimplexTree(G):
    nodes = list(G.nodes)
    edges = list(G.edges)
    st = gd.SimplexTree()
    for node in nodes:
        st.insert([node], filtration = node)
    for edge in edges:
        st.insert([edge[0], edge[1]], filtration = max(edge[0], edge[1])+epsilon)
    return(st)

#Produces copy of diagram after removing slight off-diagonal elements (most of which are created in the lower star filtration)
def removeImperfections(diagram):
    return [elt for elt in diagram if abs((elt[1][1] - elt[1][0])-epsilon) > tol]

#%%
#partitionGraph function splits large network into several networks, one for each community
#Parameters:
    #graph: of type nx.Graph
    #partition: dictionary of {node:partition number}
    #numClusters: total number of clusters in the partition (saves time if known)
    #threshold: if true, removes all subgraphs with nodes fewer than this value
#Returns:
    #partitionGraphs: list of subgraphs, element i being the subgraph with partition value i,
    #numNodes: list of numbers of nodes in each subgraph.
    #numEdges: list of numbers of edges in each subgraph.
def partitionGraph(graph, partition, numClusters = None, threshold = 10):
    startTime = time.perf_counter()
    if numClusters is None:
        numClusters = numBPClusters(partition, silent = False)
    print('making lists...')
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    partitionGraphs = []
    for i in range(numClusters):
        partitionGraphs.append(nx.Graph())
    numNodes = [0]*numClusters
    numEdges = [0]*numClusters
    print('done')
    for i, pair in enumerate(edges):
        if i % 100000 == 0:
            print('checking pair ' + str(i))
        part0 = partition[int(pair[0])]
        part1 = partition[int(pair[1])]
        if part0 == part1:
            partitionGraphs[part0].add_edge(pair[0], pair[1])
            numEdges[part0] = numEdges[part0] + 1
    for i, node in enumerate(nodes):
        if i % 10000 == 0:
            print('checking node ' + str(i))
        pnode = partition[int(node)]
        partitionGraphs[pnode].add_node(node)
        numNodes[pnode] = numNodes[pnode] + 1
    if threshold != 0:
        print('removing subgraphs of degree below threshold')
        cutGraphs = []
        cutNodes = []
        cutEdges = []
        for i, graph in enumerate(partitionGraphs):
            if numNodes[i] >= threshold:
                cutGraphs.append(graph)
                cutNodes.append(numNodes[i])
                cutEdges.append(numEdges[i])
        partitionGraphs = cutGraphs
        numNodes = cutNodes
        numEdges = cutEdges
    print('number of (threshold+ size) communities in graph: ' + str(len(numNodes)))
    print('function time in seconds: ' + str(time.perf_counter() - startTime))
    return [partitionGraphs, numNodes, numEdges]

#%%
#getPersistence function.
#Converts each graph in list partitiongraph to a gudhi SimplexTree via lower star filtration.
#If plot is True, also produces persistence diagram and barcode plots.
#Returns corresponding list of persistence diagrams. Also returns list of SimplexTrees if returnSC is True
#Warning! Plotting network diagrams is very slow!
def getPersistence(partitiongraph, returnSC = False, display = 0):
    persistenceDiagrams = list()
    if returnSC == True:
        simplexTrees = list()
    for i, subgraph in enumerate(partitiongraph[0]):
        if i % 100 == 0:
            print('Creating PDs for partition ' + str(i))
        simplexTree = toSimplexTree(subgraph)
        persistenceDiagram = removeImperfections(simplexTree.persistence(homology_coeff_field=2))
        persistenceDiagrams.append(persistenceDiagram)
        if returnSC == True:
            simplexTrees.append(simplexTree)
        if i < display:
            avgDeg = "{:.2f}".format(partitiongraph[1][i]/partitiongraph[2][i])
            gd.plot_persistence_diagram(persistenceDiagram)
            plt.title('PD for subnetwork ' + str(i) + '. Nodes: ' + str(partitiongraph[1][i]) + '. Edges: ' + str(partitiongraph[2][i]) + '. Average degree: ' + avgDeg)
            plt.show()
            nx.draw(subgraph, node_size = 100)
            plt.title('Network diagram for subnetwork ' + str(i) + '. Nodes: ' + str(partitiongraph[1][i]) + '. Edges: ' + str(partitiongraph[2][i]) + '. Average degree: ' + avgDeg)
            plt.show()
            #gd.plot_persistence_barcode(persistenceDiagram)
            #plt.title('PB for subnetwork ' + str(i) + '. Nodes: ' + str(partitiongraph[1][i]) + '. Edges: ' + str(partitiongraph[2][i]))
            #plt.show()
    if returnSC == True:
        return persistenceDiagrams, simplexTrees
    if returnSC == False:
        return persistenceDiagrams

#%%

#Create dataset class for PIs
#Following https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class PIsDataset(Dataset):
    """Persistence Images dataset."""

    def __init__(self, data_source, transform=None):
        self.PIs = data_source
        self.transform = transform

    def __len__(self):
        return len(self.PIs)

    def __getitem__(self, idx):
        sample = self.PIs[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample
        
#%%
#PIsfromPDs function. Produces persistence images from image dataset or point cloud (ripsVersion).
#note: Uses deprecated PersImage class. See Application III for use of new PersistenceImager class.
#Parameters:
    #persistenceDiagrams: list of persistence diagrams.
    #spread: Standard deviation of Gaussian function in the PI trnasformation. Unlikely to affect accuracy.
    #pixels: Output size of persistence image.
    #display: number of PDs/PIs to display (displays first <display> images).
    #countDuds: if True, prints percentage of "problematic" diagrams that contain only one H_0 class (see Figure 15 of the paper for an example).
#Returns:
    #output: list of ndarrays (PIs) with shape = shape(pixels).
def PIsFromPDs(persistenceDiagrams, spread = 1, pixels = [28,28], display = 0, countDuds = False):
    output = []
    counter = 0
    for i, dgm in enumerate(persistenceDiagrams):
        if i % 100 == 0:
            print('doing image number: ' + str(i))
        birthDeathPairs = [tupl[1] for tupl in dgm]
        dgm2 = [tupl for tupl in birthDeathPairs if np.isinf(tupl[1]) == False]
        if countDuds == True:
            if len(dgm2) == 0:
                counter = counter + 1
        pim = persim.PersImage(spread=spread, pixels=pixels, verbose=False)
        persimg = pim.transform(dgm2)
        output.append(persimg)
        if i < display:
            plot_diagrams(dgm2)
            plt.title('Persistence Diagram index: ' + str(i))
            plt.show()
            pim.show(persimg)
            plt.title('Persistence Image index: ' + str(i))
            plt.show()
    if countDuds == True:
        print('Diagrams with only one class: ' + str(counter) + '/' + str(len(persistenceDiagrams)) + '. Percentage of diagrams with only 1 class: ' + str(100*counter/len(persistenceDiagrams)))
    return(output)

#%%
#normalise data, convert to tensor and add dimension of size 1
def normalisePIs(persistenceImages):
    PImean = np.mean(persistenceImages)
    PIstd = np.std(persistenceImages)
    return list(map(lambda x: torch.tensor(((x - PImean)/PIstd).copy()).unsqueeze(0), persistenceImages))

#%%
#Augment function. Converts list of tensors (data) and list of target values into list of lists with elements (data, target value).
def augment(data, target_values):
     output = []
     for i, datapoint in enumerate(data):
         output.append([datapoint.float(), target_values[i]])
     return output
 
#%%

#NN hyperparameters
n_epochs = 20
batch_size_train = 64
batch_size_test = 100
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
#%%
#Takes list of PI data, each element a list of (datapoint, target_value), test size and train size, and returns two Dataset classes of 
def dataToDatasets(data_list, test_size = 0.2, random_state = random_seed):
    flat_datalist = [item for sublist in data_list for item in sublist] #expand list of lists to flat list, each entry  a datapoint
    x = [elt[0] for elt in flat_datalist]
    y = [elt[1] for elt in flat_datalist]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state) 
    
    train_data = augment(x_train, y_train)
    test_data = augment(x_test, y_test)

    train_dataset = PIsDataset(train_data)
    test_dataset = PIsDataset(test_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)
    
    return train_loader, test_loader

#%%

#LoadData pipeline function. Takes a list of strings of graph names and produces list of augmented, normalised PIs,
#then loads this into PyTorch train_loader and test_loader
#spreads: list of standard devs to use in Gaussian for PI transformation. Results likely insensitive to this value. Default 200 for all
#display: Number of network diagrams/PDs/PIs to show in each iteration
#test_size: fraction/number of points in dataset to take as test data (same style as sklearn train_test_split).
def LoadData(graphNames, graphs = None, partitions = None, spreads = [200, 200], display = 0, test_size = 0.2):
    data_list = []
    addToSpreads = 0
    if spreads == []:
        addToSpreads = 1
    for i, graphName in enumerate(graphNames):
        print("loading graph data with index " + str(i))
        if addToSpreads == 1:
            spreads.append(200)
        if partitions == None:
            print("partitioning...")
            graph, partition = BestPartition(graphName)
        else:
            graph = graphs[i]
            partition = partitions[i]
        print("applying partitiongraph...")
        partitionedGraph = partitionGraph(graph, partition)
        print("getting persistence...")
        PDs = getPersistence(partitionedGraph, returnSC = False, display = display)
        print("getting PIs and normalising...")
        PIsNormalised = normalisePIs(PIsFromPDs(PDs, spread = spreads[i], display = display))
        PIs_data = augment(PIsNormalised, [i]*len(PIsNormalised))
        data_list.append(PIs_data)
    print("made data list. Converting to loaders...")
    train_loader, test_loader = dataToDatasets(data_list = data_list, test_size = test_size)
    return data_list, train_loader, test_loader

#%%
print("Loading and partitioning graphs. This will take some time.")
try:
    graph2, partition2 = BestPartition("com-youtube.ungraph.txt")
    graph4, partition4 = BestPartition("roadNet-PA.txt")
except FileNotFoundError:
    print("Couldn't find a graph .txt file. Did you add com-youtube.ungraph.txt and roadNet-PA.txt to your working directory?")
data_list, train_loader, test_loader = LoadData(["com-youtube.ungraph.txt", "roadNet-PA.txt"], graphs = [graph2, graph4], partitions = [partition2, partition4], spreads = [200, 200])

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
        return F.log_softmax(x)
    
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
ax2.legend(['Test Accuracy'], loc=[0.678,0.33])
fig.tight_layout()
plt.show()
