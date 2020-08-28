import math
import numpy as np
import pandas as pd
import torch
from numpy.core._multiarray_umath import ndarray
from torch import zeros
from torchvision import datasets, transforms
import csv
import time

### Prepere dataset: Input feature value and output feature valie of training dataset in
### trainset_array (number_of_sample *number_of_feature format)and trainset_labels 
### respectively  
### Input feature value and output feature valie of testing dataset in
### testset_array (number_of_sample *number_of_feature format)and testset_labels 
### respectively


# reading the isolet dataset
print("\nreading data...")
start_time1 = time.time()
csvfile=open('isolet_csv.csv')
readCSV=pd.read_csv(csvfile)
print("reading time =--- %s seconds ---" % (time.time() - start_time1))

#dividing data into training and testset and labels and reshaping 
print("\nSeperating data into training and test array and labels ...")
start_time = time.time()
X=[]
y=readCSV['class']
for i in list(readCSV.keys()):
    if i!='class':
        x=readCSV[i]
        X.append(x)
NS=readCSV.shape[0]
NF=readCSV.shape[1]-1
X=np.transpose(X)
X=np.array(X)
Y=[]
for k in y:
    for i in range (1,27):
        if k=="'{}'".format(i):
            Y.append(i)
            break
            
            

Numsubject=150
partition=0.8
nClass=np.size(np.unique(Y))
nSample=2
trainset_array=X[0:int(Numsubject*partition*nClass*nSample)]
testset_array=X[int(Numsubject*partition*nClass*nSample):]
trainset_labels=Y[0:int(Numsubject*partition*nClass*nSample)]
testset_labels=Y[int(Numsubject*partition*nClass*nSample):]
print("Reshaping time =--- %s seconds ---" % (time.time() - start_time))


# creating level hypervector
print("\ncreating level hypervector...")
starttime=time.time()

lMax = np.max(trainset_array)
lMin = np.min(trainset_array)

M = 11  # found to have best accuracy with isolet in matlab

dL = (lMax - lMin) / M
L=np.linspace(lMin,lMax,M)


nLevel = np.size(L)
D = 10000
LD = np.zeros((nLevel, D))

LD[0,] = np.random.randint(0,2,[1, D])

i = 1
j = 2
indAlter = list(range(0,D))
np.random.shuffle(indAlter)
for j in range(1,nLevel):
    LD[j, ] = LD[j - 1, ]
    nAlter=int(D/nLevel)+1
    for i in range(0,nAlter):
        k=indAlter[i+(j-1)*nAlter]
        LD[j][k] = int(not(LD[j][k])) # does nothing??
LD=LD.astype(int)
print("time to create level hypervector =--- %s seconds ---" % (time.time() - start_time))


# creating identity hypervector
print("\ncreating identity hypervector...")
start_time=time.time()
xtrainset = trainset_array.shape
# print(xtrainset)

numSamples = xtrainset[0]
numFeatures = xtrainset[1]

ID = np.random.randint(0,2,[numFeatures, D])
ID=ID.astype(int)
print("time to create identity hypervector =--- %s seconds ---" % (time.time() - start_time))


# creating class hypervector

start_time = time.time()

print("\ntraining: creating class hypervector...")

classes_ = np.unique(trainset_labels)
nClass = np.size(classes_)

# print(nClass)

CH = np.zeros((nClass, D),int)

i_ = 1
j__ = 1
trainset_array=trainset_array.reshape(numSamples,numFeatures)

# print(numSamples)
# print(nLevel)

for i in range(numSamples):
    sHV = np.zeros((1, D),int)
    for j in range(numFeatures):
        indLevel = np.argmin(abs(L - trainset_array[i, j] * np.ones(nLevel)))
        sHV = (sHV+np.bitwise_xor(LD[indLevel, ], ID[j, ]))
    sBundled = (sHV > int((numFeatures / 2)))

    sBundled=sBundled.astype(int)
    for k in range(np.size(classes_)):
        if trainset_labels[i] == classes_[k]:
            CH[k, ] = (CH[k, ] + sBundled)
            break

from collections import Counter
GR = Counter(trainset_labels)
nClass = np.size(classes_)
classHV = np.zeros((nClass, D),int)

for i in range(nClass):
    classHV[i, ] = ((CH[i, ] >= (GR[classes_[i]] / 2)))
    classHV[i,]=classHV[i,].astype(int)
print("training time =--- %s seconds ---" % (time.time() - start_time))

#Testing
print("\ntesting....")
start_time=time.time()
xtestset = testset_array.shape
numTestSample = xtestset[0]
numTestFeatures = xtestset[1]

testset_array=testset_array.reshape(numTestSample,numTestFeatures)

HDC_label=[]
accuracy=0
for i in range(numTestSample):
    
    sHV = np.zeros((1, D),int)
    for j in range(numFeatures):
        indLevel = np.argmin(abs(L - testset_array[i, j] * np.ones(nLevel)))
        sHV = (sHV+np.bitwise_xor(LD[indLevel, ], ID[j, ]))
    QH = (sHV >= (numFeatures / 2))
    QH = QH.astype(int)
    Test_V=[]    
    for k in range(np.size(classes_)):
        Test_V.append(sum(np.bitwise_xor(QH[0],classHV[k])))
    HDC_label.append(classes_[np.argmin(Test_V)])
    if HDC_label[i]==testset_labels[i]:
        accuracy=accuracy+1
accuracy=accuracy/numTestSample
print('accuracy={}%'.format(accuracy*100))
print("time for testing =--- %s seconds ---" % (time.time() - start_time))
print("Total Runtime=--- %s seconds ---" % (time.time() - start_time1))


