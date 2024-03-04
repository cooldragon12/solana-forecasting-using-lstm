from torch import nn

class SolanaLSTMModel(nn.Module):
    """
    LSTM model for Forecasting

    Using Pytorch
    """
    #LSTM model for Forecasting 
    def __init__(self, input_size, hidden_size, num_layers):
        super(SolanaLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden,_)= self.lstm(x)
        out = self.fc(hidden[-1])
        return out