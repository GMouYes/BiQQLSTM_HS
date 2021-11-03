import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
import time
import math, copy
from mask import mask_softmax, mask_mean, mask_max
from transformers import BertModel

from qqrnn import BiQQLSTM,BiQQLSTMConf


def save_model(model, model_path, grid):
    """Save model."""
    torch.save(model.state_dict(), model_path)
    # torch.save(model.module.state_dict(), model_path)
    with open("hyper.pkl",'wb') as f:
        pickle.dump(grid,f)
    #print("checkpoint saved")
    return

def readGrid():
    with open("hyper.pkl",'rb') as f:
        grid = pickle.load(f)
    return grid

def load_model(model, model_path):
    """Load model."""
    map_location = 'cpu'
    if torch.cuda.is_available():
        map_location = 'cuda:0'
    model.load_state_dict(torch.load(model_path, map_location))
    return model

def get_model_setting(**args):
    """Load Model Settings"""
    model = fairness(**args)
    return model


class myBiQQLSTM(nn.Module):
    """docstring for TextLSTM"""
    def __init__(self, **args):
        super(myBiQQLSTM, self).__init__()

        self.myconf = BiQQLSTMConf(
            input_dims=[[args["inputDim"]]],
            hidden_dim=args["lstm_out"],
            dropout=args['lstm_dropout'],
            num_layers=args['lstm_layers'],
            window=args["window_size"],
            )
        # [B, H, D]
        self.BiQQLSTM = BiQQLSTM(self.myconf)

    def forward(self, x):
        # self.BiQRNN.flatten_parameters()
        BiQQLSTM_out = self.BiQQLSTM(x)
        return BiQQLSTM_out

class fullyConnected(nn.Module):
    """docstring for warping"""
    def __init__(self, **args):
        super(fullyConnected, self).__init__()
        self.lstm_out = 2*args["lstm_out"]
        # self.lstm_out = 2*args["inputDim"]

        self.attentionLinear = nn.Linear(self.lstm_out, 1)

        self.linear_out1 = args["linear_out1"]
        self.linear_out2 = args["linear_out2"]

        self.dropout1 = args["dropout1"]
        self.dropout2 = args["dropout2"]

        self.output_size = args["output_size"]
        
        self.fc = nn.Sequential(
            nn.Linear(3*self.lstm_out, self.linear_out1),
            nn.ReLU(),
            nn.Dropout(self.dropout1),

            nn.Linear(self.linear_out1, self.linear_out2),
            nn.ReLU(),
            nn.Dropout(self.dropout2),

            nn.Linear(self.linear_out2, self.output_size),
            #nn.Sigmoid(),
            )

    def forward(self, x):
        attn = self.attentionLinear(x).squeeze(-1)
        attn = mask_softmax(attn)
        attn = torch.sum(attn.unsqueeze(-1) * x, dim=1) # (b,h)

        max_pool = mask_max(x)
        avg_pool = mask_mean(x)

        result = torch.cat([attn,max_pool,avg_pool], dim=1)
        return self.fc(result)

        
class myModel(nn.Module):
    """docstring for myModel"""
    def __init__(self, **args):
        super(myModel, self).__init__()
 
        self.BiQQLSTM = myBiQQLSTM(**args)
        self.fc = fullyConnected(**args)

    def forward(self, x):
        
        x = self.BiQQLSTM(x)
        x = self.fc(x)
        return x

class fairness(nn.Module):
    """docstring for fairness"""
    def __init__(self, **args):
        super(fairness, self).__init__()

        self.embedder = BertModel.from_pretrained('bert-base-uncased')
        self.model = myModel(**args)

    def forward(self, x):
        x0,x1 = x
        x0 = self.embedder(**x0)[0]
        x0 = self.model(x0)

        x1 = self.embedder(**x1)[0]
        x1 = self.model(x1)

        return x0, x1