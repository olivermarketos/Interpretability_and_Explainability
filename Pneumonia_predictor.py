from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import torch
import pandas as pd


class PneumoniaPredictorCNN(nn.Module):
    def __init__(self, image_size=(128, 128),dropout=0.5):

        super().__init__()
        self.image_size = image_size

        self.layers = nn.ModuleList()


        self.layers.append(nn.Conv2d(in_channels=1, out_channels= 32 , kernel_size= 5))
        self.layers.append(nn.BatchNorm2d(32))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layers.append(nn.Conv2d(in_channels=32, out_channels= 64 , kernel_size= 5))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU())     
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
   

        self.layers.append(nn.Conv2d(in_channels=64, out_channels= 96 , kernel_size= 5))
        self.layers.append(nn.BatchNorm2d(96))
        self.layers.append(nn.ReLU())     
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))


        self.layers.append(nn.Conv2d(in_channels=96, out_channels= 128 , kernel_size= 5))
        self.layers.append(nn.BatchNorm2d(128))
        self.layers.append(nn.ReLU())    
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
 

        self.layers.append(nn.Conv2d(in_channels=128, out_channels= 160 , kernel_size= 5))
        self.layers.append(nn.BatchNorm2d(160))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layers = nn.Sequential(*self.layers)

        # calculate the output size after the conv layers
        num_conv_features, h_out, w_out = self.calc_conv_output_size(self.image_size)
        self.flattened_size = num_conv_features * h_out * w_out
        print(f"Flattened size after conv layers: {self.flattened_size}  {num_conv_features} channels x {h_out} H x {w_out} W)")

        self.fc = nn.Sequential(nn.Linear(self.flattened_size, 128),
                                nn.ReLU(),                            
                                nn.BatchNorm1d(128),
                                nn.Dropout(dropout)
        )

        self.fc2 = nn.Linear(128, 2)

    def calc_conv_output_size(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size) # batch 1, channels 1
            dummy_output = self.layers(dummy_input)
            output_shape = dummy_output.shape
            return output_shape[1], output_shape[2], output_shape[3] # (channels, height, width)
        
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.fc2(x)
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