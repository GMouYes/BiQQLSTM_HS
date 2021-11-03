# pytorch
from torch.utils.data import Dataset
import torch
# from torchtext import data
# from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler

class myDataset(Dataset):
    ''' dataset reader
    '''
    
    def __init__(self, X, y):
        self.X, self.y = X, y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        data = self.X[index]
        # print(type(data),data)
        y = torch.tensor(self.y[index],dtype=torch.float)
        
        return data, y

def readData(datatype:str):
    path = "../data/"
    # augment_ratio = 0.6

    text = pd.read_csv(path+'{}.csv'.format(datatype))
    label = text['class'].to_numpy().reshape(-1,1)

    # if datatype=='train':
    #     hate_shape = int((text['class']==1).sum() *augment_ratio)
    #     legit_shape = int((text['class']==0).sum() *augment_ratio)
    #     hateAug = pd.read_csv(path+'mediumNlgHateReadibilityColaAug.csv', nrows=hate_shape)
    #     legitAug = pd.read_csv(path+'mediumNlgLegitReadibilityColaAug.csv', nrows=legit_shape)
    #     text = list(text['fullClean']) + list(hateAug['text']) + list(legitAug['text'])
    #     hateLabel = np.ones((hate_shape,1))
    #     legitLabel = np.zeros((legit_shape,1))
    #     label = np.concatenate([label,hateLabel, legitLabel], axis=0)
    # else:
    #     text = list(text['fullClean'])


    return list(text['fullClean']), label
