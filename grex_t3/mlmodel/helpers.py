import torch
from torch.utils.data import TensorDataset, DataLoader

import os 

def makedir(path): 
	if not os.path.exists(path):
		os.makedirs(path)   

def load_data(train_dataset,labels=None,batch_size=128):

    tensor_x = torch.Tensor(train_dataset) # transform to torch tensor
    train_dataset = TensorDataset(tensor_x)

    if labels is not None:
      tensor_y = torch.Tensor(labels)
      train_dataset = TensorDataset(tensor_x,tensor_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, train_loader