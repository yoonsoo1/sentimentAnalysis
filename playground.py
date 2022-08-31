# %%
from baseDataLoader import processTxtAsCsv, customDataSet
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt

# %% 
trainPath = "data/train.csv"
testPath = "data/test.csv"
batchSize = 4

def trainValDataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

# %%
processTxtAsCsv(trainPath, testPath)
trainSet = customDataSet(trainPath)
testSet = customDataSet(testPath)

trainLengths = []
for idx in range(25000):
    getLen = len(trainSet[idx][0])
    trainLengths.append(getLen)

plt.hist(trainLengths)
plt.title("Train Set Token Lengths")
plt.show()

print("Mean: ", sum(trainLengths)/len(trainLengths))

# %%

testLengths = []
for idx in range(25000):
    getLen = len(trainSet[idx][0])
    testLengths.append(getLen)

plt.hist(testLengths)
plt.title("Test Set Token Lengths")
plt.show()

print("Mean: ", sum(testLengths)/len(testLengths))

# Let's cut the token lengths at 110 because that's the mean
# %%
from collections import Counter

allWords = []
for idx in range(len(trainSet)):
    allWords += trainSet[idx][0]
print("total word count: ", len(allWords))
countWords = Counter(allWords)
allWordCount = len(countWords)
print("distinct word count: ", allWordCount)
print(countWords)

# %%
