import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    CNN model with two fully connected layer
    """
    def __init__(self, num_conv_layers=3, base_channels=32, num_classes=10):
        super(CNN, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        in_channels = 3
        cifar_image_size = 32
        
        for i in range(num_conv_layers):
            out_channels = base_channels * (2 ** i)
            print(out_channels)
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            in_channels = out_channels
        
        self.pool = nn.MaxPool2d(2, 2)
        
        final_size =  cifar_image_size // (2 ** num_conv_layers)
        self.fc1 = nn.Linear(out_channels * final_size * final_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.pool(F.relu(conv_layer(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x