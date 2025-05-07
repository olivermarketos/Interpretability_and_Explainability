from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import torch
import pandas as pd


class PneumoniaPredictorCNN(nn.Module):
    def __init__(self, image_size=(224, 224),
                in_channels=1, 
                conv_defs=None, 
                fc_defs=None,
                fc_batch_norm=True,
                fc_dropout=0.5):

        super().__init__()
        self.image_size = image_size
        

        conv_layers = []
        current_channels = in_channels
        for layer_config in conv_defs:
            conv_layers.append(nn.Conv2d(
                in_channels=current_channels,
                out_channels=layer_config['out_channels'],
                kernel_size=layer_config['kernel_size'],
                stride=layer_config['stride'],
                padding=layer_config['padding']
                ))
            if layer_config.get('batch_norm', True):
                conv_layers.append(nn.BatchNorm2d(layer_config['out_channels']))
            conv_layers.append(nn.ReLU())
            
            if 'dropout' in layer_config and layer_config['dropout'] > 0:
                conv_layers.append(nn.Dropout2d(p=layer_config['dropout'])) # Use Dropout2d for feature maps
            if 'pool' in layer_config:
                pool = layer_config['pool']
                conv_layers.append(getattr(nn, pool['type'])(
                    kernel_size=pool.get('kernel_size', 2),
                    stride=pool.get('stride', 2)
                ))


            current_channels = layer_config['out_channels']

        self.conv = nn.Sequential(*conv_layers)

        # compute flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *self.image_size)
            out = self.conv(dummy)
            flat_features = out.numel() // out.size(0)  # numel() returns total number of elements in the tensor


        # Build FC layers
        fc_layers = []
        prev = flat_features
        for hidden in fc_defs[:-1]:
            fc_layers.append(nn.Linear(prev, hidden))
            fc_layers.append(nn.ReLU())
            if fc_batch_norm:
                fc_layers.append(nn.BatchNorm1d(hidden))
            fc_layers.append(nn.Dropout(fc_dropout))
            prev = hidden
        fc_layers.append(nn.Linear(prev, fc_defs[-1]))  # last layer without activation
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x   

   
        



class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with 'filepath' and 'label_encoded' columns.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = dataframe['filepath'].values
        self.labels = dataframe['encoded_label'].values
        self.transform = transform        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        return image, label