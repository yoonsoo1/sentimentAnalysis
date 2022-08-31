import torch.nn as nn

class lstmModel(nn.Module):
    def __init__(self, inputDim, embedDim, hiddenDim, outputDim):
        super(lstmModel, self).__init__()

        self.embedding = nn.Embedding(inputDim, embedDim)
        
        self.lstm = nn.LSTM(embedDim, hiddenDim)
        
        self.fc = nn.Linear(hiddenDim, outputDim)



    def forward(self, batch):
        batch = batch.transpose(0,1)
        embedding = self.embedding(batch)

        output, (hidden, cell) = self.lstm(embedding)
        
        result = self.fc(hidden.squeeze(0))

        return result