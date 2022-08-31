import pandas as pd
import os
import csv
from tqdm import tqdm
from torch.utils.data import Dataset
import re, string
import numpy as np
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
SIZE = 118
FILLER = 0

def processTxtAsCsv(trainPath, testPath):
    if not os.path.exists(trainPath):
        txtToCsv("data/aclImdb/test", "train")
    if not os.path.exists(testPath):
        txtToCsv("data/aclImdb/train", "test")

def txtToCsv(path, type):
    csvName = "data/" + type + ".csv"
    csvHeader = ["id", "text", "isPos", "fileName"]
    id = 0
    with open(csvName, "w") as file:
        writer = csv.writer(file)
        writer.writerow(csvHeader)
        posFilesPath = path + "/pos"
        negFilesPath = path + "/neg"
        posPbar = tqdm(os.listdir(posFilesPath))
        negPbar = tqdm(os.listdir(negFilesPath))
        for posFile in posPbar:
            with open(posFilesPath + "/" + posFile) as f:
                textString = f.readlines()
                writer.writerow([id, textString, 1, posFile])
                id += 1
                posPbar.set_description(f"Processing Positive Text Files for {type}")
        for negFile in negPbar:
            with open(negFilesPath + "/" + negFile) as f:
                textString = f.readlines()
                writer.writerow([id, textString, 0, negFile])
                id += 1
                negPbar.set_description(f"Processing Negative Text Files for {type}")

class customDataSet(Dataset):
    def __init__(self, csvPath, wordToInt):
        self.df = pd.read_csv(csvPath)
        self.df["text"] = self.df["text"].apply(self.processText)
        self.df["text"] = self.df["text"].str.split()
        self.wordToInt = wordToInt
        self.df["text"] = self.df["text"].apply(self.transTextToInt)
        self.df["text"] = self.df["text"].apply(lambda x: \
            x[:SIZE] + [FILLER]*(SIZE-len(x)))
        self.df["text"] = self.df["text"].apply(lambda x: np.array(x))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx,1]
        isPos = self.df.iloc[idx,2]
        return text, isPos

    def processText(self, text):
        text = text.lower()
        text = re.sub("<.*?>", "", text)
        text = "".join([l for l in text if l not in string.punctuation])
        text = [word for word in text.split() if word not in stops]
        text = " ".join(text)
        return text

    def transTextToInt(self, text):
        newText = []
        for word in text:
            if word in self.wordToInt:
                newText.append(self.wordToInt[word])
            else:
                newText.append(1) # 1 represents unknown
        return newText