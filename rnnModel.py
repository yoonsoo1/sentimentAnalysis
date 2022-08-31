import torch.nn as nn

class rnnModel(nn.Module):
    def __init__(self, inputDim, embedDim, hiddenDim, outputDim):
        super(rnnModel, self).__init__()

        self.embedding = nn.Embedding(inputDim, embedDim)
        
        self.rnn = nn.RNN(embedDim, hiddenDim)
        
        self.fc = nn.Linear(hiddenDim, outputDim)
        
    def forward(self, batch):
        batch = batch.transpose(0,1)
        embedding = self.embedding(batch)

        output, hidden = self.rnn(embedding)
        
        # Many-to-one RNN so we don't use the output
        result = self.fc(hidden.squeeze(0))

        return result