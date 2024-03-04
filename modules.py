import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class SolanaDataModule(pl.LightningDataModule):
    """This class is a PyTorch Lightning DataModule that will handle the data loading and possible preprocessing."""
    def __init__(self, train_sequences, test_sequences, batch_size=32):
        super().__init__()
        self.train_sequences = train_sequences
        self.val_sequences = test_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        self.train = SolanaDataset(self.train_sequences)
        self.val = SolanaDataset(self.val_sequences)
        self.test = SolanaDataset(self.test_sequences)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, shuffle=False, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1, shuffle=False, num_workers=0)


class SolanaDataset(torch.utils.data.Dataset):
    """This class is a PyTorch Dataset that will handle the sequences and labels as Tensor Datasets."""
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, label = self.sequences[idx]

        return dict(
            sequence=torch.Tensor(seq.to_numpy()),
            label=torch.tensor(label).float()
        )
    

