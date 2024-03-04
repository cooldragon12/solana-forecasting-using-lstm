from torch import nn
import torch
import pytorch_lightning as pl

from model import SolanaLSTMModel


class SolanaPricePredictor(pl.LightningModule):
    """This class is a PyTorch Lightning Module will handle the training, validation, and testing of the LSTM model."""
    def __init__(self, n_features:int, hidden_size:int, num_layers:int):
        super().__init__()
        self.model = SolanaLSTMModel(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers)
        self.loss = nn.MSELoss()
    
    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        
        if labels is not None:
            loss = self.loss(output, labels)
        
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']

        loss, outputs = self(sequences, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']

        loss, outputs = self(sequences, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']

        loss, outputs = self(sequences, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer  