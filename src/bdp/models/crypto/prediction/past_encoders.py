import torch
from torch import nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, output_dim,**kwargs):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num

        # The LSTM takes input sequences and outputs hidden states
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_num, batch_first=True)
        
        # The linear layer that maps from hidden state space to output space
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self,packed_input):
        # Pack the input batch
        #packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Forward pass through LSTM layer
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # You might need to unpack the output or directly use the hidden state to make predictions
        # For simplicity, let's use the last hidden state
        output = self.linear(hidden[-1])
        return output
    
