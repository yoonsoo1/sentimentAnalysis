from lib2to3.pgen2.tokenize import tokenize
from baseDataLoader import processTxtAsCsv, customDataSet
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_convert
from sklearn.model_selection import train_test_split
import torch
from rnnModel import rnnModel
from lstmModel import lstmModel
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm 
import numpy as np
import pandas as pd
from collections import Counter
import os, shutil

""" 
Improvements I can make:
    1. Change the data processing to truncate at the beginning 
    and pad at the beginning: currently it does it at the end

    2. Changed the learning rate because the rate at which
    it learned with ADAM was slow (and we usually used 0.01 for
    ADAM not 0.001)

    3. Can fiddle with the batch_size. We have in the past 
    experienced big batch sizes negatively impacting learning.

    4. Unk token unseen during training because dataloader doesnt load it.   
"""

trainPath = "data/train.csv"
testPath = "data/test.csv"
batchSize = 128 # For RNN
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
epochs = 10
valEvery = 5 #How often validation occurs in training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trainValDataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets["train"] = Subset(dataset, train_idx)
    datasets["val"] = Subset(dataset, val_idx)
    return datasets

def setUpRNN(wordToInt):
    calcInputDim = len(wordToInt)+2
    model = rnnModel(inputDim=calcInputDim, embedDim=EMBEDDING_DIM, hiddenDim=HIDDEN_DIM, outputDim=1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = nn.BCEWithLogitsLoss()
    model = model.to(device)
    loss = loss.to(device)
    return model, optimizer, loss

def setUpLSTM(wordToInt):
    calcInputDim = len(wordToInt)+2
    model = lstmModel(inputDim=calcInputDim, embedDim=EMBEDDING_DIM, hiddenDim=HIDDEN_DIM, outputDim=1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = nn.BCEWithLogitsLoss()
    model = model.to(device)
    loss = loss.to(device)
    return model, optimizer, loss

def getBinAcc(prediction, actual, activation="tanh"):
    roundPred = None
    if activation == "tanh":
        roundPred = torch.round(torch.tanh(prediction))
    elif activation == "sigmoid":
        roundPred = torch.round(torch.sigmoid(prediction))
    correct = (roundPred == actual).float()
    binAcc = correct.sum() / len(correct)
    return binAcc.item()

def genWordToInt(path):
    df = pd.read_csv(path)
    corpus = [word for text in df["text"] for word in text.split()]
    countWords = Counter(corpus)
    sortedWords = countWords.most_common()
    # 0: pad, 1: unknown
    wordToInt = {word:i+2 for i, (word, count) in enumerate(sortedWords)}
    return wordToInt

def getIterators():
    # Split train and val from training data
    wordToInt = genWordToInt(trainPath)
    trainSet = customDataSet(trainPath, wordToInt)
    splitSet = trainValDataset(trainSet)
    trainLoader = DataLoader(splitSet["train"], shuffle=True, \
        batch_size=batchSize)
    valLoader = DataLoader(splitSet["val"], shuffle=True, \
        batch_size=batchSize)
    # Load Test Set
    testSet = customDataSet(testPath, wordToInt)
    testSetLen = len(testSet)
    testLoader = DataLoader(testSet, shuffle=True,\
        batch_size=batchSize)
    return trainLoader, valLoader, testLoader, wordToInt, testSetLen

def train(model, trainIt, valIt, optimizer, loss):
    bestLoss = float("inf")
    bestBinAcc = float("-inf")
    for epoch in tqdm(range(epochs), desc="Running Training"):
        model.train()
        
        for inputs, labels in trainIt:
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            prediction = model(inputs).squeeze(1)
            binAcc = getBinAcc(prediction, labels)
            # print(f"Binary Accuracy for training epoch {epoch}: ", binAcc)
            trainLoss = loss(prediction, labels.float())
            trainLoss.backward()
            # print(f"Loss for training epoch {epoch}: ", trainLoss.item())
            optimizer.step()

        if epoch%valEvery == 0:
            model.eval()
            for inputs, labels in valIt:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                prediction = output.squeeze(1)
                binAcc = getBinAcc(prediction, labels)
                valLoss = loss(prediction, labels.float()).item()

                if binAcc > bestBinAcc:
                    torch.save(model.state_dict(), "bestModels/bestBinAcc.pt")
                    print("Best BinAcc Achieved at ", binAcc, " (prev: ", bestBinAcc, ")")
                    bestBinAcc = binAcc

                if valLoss < bestLoss:
                    torch.save(model.state_dict(), "bestModels/bestLoss.pt")
                    print("Best Loss Achieved at ", valLoss, " (prev: ", bestLoss, ")")
                    bestLoss = valLoss

def test(model, testIt, loss, testSetLen):
    model.eval()
    numCorrect = 0
    tp, fp, fn = 0, 0, 0
    for inputs, labels in testIt:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        prediction = torch.tanh(output.squeeze(1))
        modelPred = torch.round(prediction)

        forTP = torch.add(modelPred, labels)
        tp += torch.numel(forTP[forTP == 2])
        fp += ((1 - labels) * modelPred).sum().item()
        fn += ((1 - modelPred) * labels).sum().item()
        corrects = modelPred.eq(labels.float().view_as(modelPred))
        corrects = np.squeeze(corrects.cpu().numpy())
        numCorrect += np.sum(corrects)
    
    precision, recall, f1 = calcMetrics(tp, fp, fn)
    print("Test Binary Accuracy: {:.4f}".format(numCorrect/testSetLen))
    print("Test Precision: {:.4f}".format(precision))
    print("Test Recall: {:.4f}".format(recall))
    print("Test F1: {:.4f}".format(f1))

def calcMetrics(tp, fp, fn):
    precision = 0
    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    recall = 0
    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    f1 = 0
    if (precision+recall) != 0:
        f1 = 2 * (precision*recall)/(precision+recall)
    return precision, recall, f1

def clearBests():
    folder = "bestModels"
    for file in os.listdir(folder):
        filePath = os.path.join(folder, file)
        try:
            if os.path.isfile(filePath) or os.path.islink(filePath):
                os.unlink(filePath)
            elif os.path.isdir(filePath):
                shutil.rmtree(file)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (filePath, e))

if __name__ == "__main__":
    processTxtAsCsv(trainPath, testPath)
    trainIt, valIt, testIt, wordToInt, testSetLen = getIterators()
    model, optimizer, loss = setUpRNN(wordToInt)
    # model, optimizer, loss = setUpLSTM(wordToInt)
    clearBests()
    train(model, trainIt, valIt, optimizer, loss)
    model.load_state_dict(torch.load("bestModels/bestBinAcc.pt"))
    test(model, testIt, loss, testSetLen)



