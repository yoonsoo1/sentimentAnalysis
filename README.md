# Sentiment Analysis for Natural Language
This is my implementation of interpreting the iMDB sentiment data with PyTorch.

## Methodology
### Pipeline
Data pipeline is built so that any NLP model can be run through with the same
data processing pipeline. With this resuseable pipeline, if one wants to change the model to use,\
all they need to do is add a function similar to setUpRNN or setUpLSTM in the main.py. \
In this set up, we must specify the model and the loss function that we want to use. 

### Project Background
The main goal of the project was to compare pros and cons of using the RNN model vs the LSTM model. \
\
With the popularity of deep learning frameworks for natural language processing, \
using the RNN model and the LSTM model is becoming more commonplace.\
However, despite LSTM solving the problem of vanishing gradient descent problem of RNNs,\
it is computationally more expensive.

## Results
The LSTM does a much better job for its binary accuracy and loss when ran for a long time on the iMDB sentiment data \
However, the RNN is a lightweight model that can run many epochs in a short amount of time. \

With the LSTM, the project achieved over 70% binary accuracy, compared to RNN at around 60%. \
There are some hyperparameter adjustments that need to be done to increase the accuracy. \
\
However, the LSTM training took 25% more in terms of computing time. \ 
In conclusion, if compute is not a big problem and we need a higher accuracy, we can use the LSTM model but with a lightweight\
RNN model, we may be able to achieve an acceptable accuracy using less compute power. 

## Limitations
The future improvements are specified as comments in main.py. 
