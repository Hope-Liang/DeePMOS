import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
    

class MeanNet(nn.Module):
    
    def __init__(self):
        super(MeanNet, self).__init__()
        self.mean_conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32,  kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.mean_rnn = nn.LSTM(input_size = 512,
                                hidden_size = 128,
                                num_layers = 1,
                                batch_first = True,
                                bidirectional = True)
        
        self.mean_MLP = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1)
        )

        self.var_MLP = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1),
            nn.ReLU()
        )
        
    def forward(self, speech_spectrum):
        # input speech_spectrum shape (batch, 1, max_seq_len, 257)
        batch = speech_spectrum.shape[0]
        time = speech_spectrum.shape[2]
        speech_spectrum = self.mean_conv(speech_spectrum) # shape (batch, 128, max_seq_len, 4)
        speech_spectrum = speech_spectrum.view((batch, time, 512)) # shape (batch, max_seq_len, 512)
        speech_spectrum, (h, c) = self.mean_rnn(speech_spectrum) # shape (batch, max_seq_len, 256)
        mos_mean = self.mean_MLP(speech_spectrum) # shape (batch, max_seq_len, 1)
        mos_var = self.var_MLP(speech_spectrum) # shape (batch, max_seq_len, 1)
        return mos_mean, mos_var


class DeePMOS(nn.Module):
    
    def __init__(self):
        super(DeePMOS, self).__init__()
        self.MeanNet = MeanNet()

    def forward(self, speech_spectrum):
        mos_mean, mos_var = self.MeanNet(speech_spectrum)
        return mos_mean, mos_var
            
            
        