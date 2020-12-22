
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import random_split


# dataset definition
class CSVDataset(Dataset):
  # load the dataset
  def __init__(self, path, num_inputs=78, type='score'):
    self.num_inputs = num_inputs
    self.type = type
    # load the csv file as a dataframe
    df = pd.read_csv(path).set_index('gameID', drop=True)

    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    # cudnn.benchmark = True

    # store the inputs and outputs
    self.X = df.values[:, 0:num_inputs].astype('float32')
    if type == 'score':
      self.y = df.values[:, num_inputs:num_inputs+2].astype('float32')
    elif type == 'ml':
      self.y = df.values[:, 'awayWin':'spread'].astype('int32')
    else:
      raise ValueError('Invalid output type \'', type, '\' given to CSVDataset!')
    # ensure target has the right shape
    self.y = self.y.reshape((len(self.y), 2))

  # number of rows in the dataset
  def __len__(self):
    return len(self.X)

  # get a row at an index
  def __getitem__(self, idx):
    return [self.X[idx], self.y[idx]]

  def concat(self, df):
    # store the new inputs and outputs
    newX = df.values[:, 0:self.num_inputs].astype('float32')
    if self.type == 'score':
      newY = df.values[:, self.num_inputs:self.num_inputs+2].astype('float32')
    elif self.type == 'ml':
      newY = df.values[:, 'awayWin':'spread'].astype('int32')
    else:
      raise ValueError('Invalid output type \'', type, '\' given to CSVDataset!')
    # add the new data onto the existing X and y
    self.X = np.concatenate((self.X, newX), axis=0)
    self.y = np.concatenate((self.y, newY), axis=0)
    # ensure target has the right shape
    self.y = self.y.reshape((len(self.y), 2))

  # get indexes for train and test rows
  def get_splits(self, n_test=0.30):
    # determine sizes
    test_size = round(n_test * len(self.X))
    train_size = len(self.X) - test_size
    # calculate the split
    return random_split(self, [train_size, test_size])
