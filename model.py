import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import pandas as pd
from boson_sampler import BosonSampler
from utils import accuracy
import perceval as pcvl
import perceval.providers.scaleway as scw  # Uncomment to allow running on scaleway

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class MnistModel(nn.Module):
    def __init__(self, device = 'cpu', embedding_size = 0):
        super().__init__()
        input_size = 28 * 28
        num_classes = 10
        self.device = device
        self.embedding_size = embedding_size
        if self.embedding_size:
            input_size += embedding_size #considering 30 photons and 2 modes
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, xb, emb = None):
        xb = xb.reshape(-1, 784)
        if self.embedding_size and emb is not None:
            # concatenation of the embeddings and the input images
            xb = torch.cat((xb,emb),dim=1)
        out = self.linear(xb)
        return(out)
    
    def training_step(self, batch, emb = None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        if self.embedding_size:
            out = self(images, emb.to(self.device)) ## Generate predictions
        else:
            out = self(images) ## Generate predictions
        loss = F.cross_entropy(out, labels) ## Calculate the loss
        acc = accuracy(out, labels)
        return loss, acc
    
    def validation_step(self, batch, emb =None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        if self.embedding_size:
            out = self(images, emb.to(self.device)) ## Generate predictions
        else:
            out = self(images) ## Generate predictions
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return({'val_loss':loss, 'val_acc': acc})
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return({'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()})
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        return result['val_loss'], result['val_acc']
    
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.lin = nn.Linear(hidden_channels, 10)

    def forward(self, data, batch=None):
        x, edge_index = data.x, data.edge_index

        # First Graph Convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second Graph Convolution
        x = self.conv2(x, edge_index)

        # Pooling layer
        batch = torch.zeros(data.x.shape[0],dtype=int) if batch is None else batch
        x = global_mean_pool(x, batch)
        print(x.shape)

        # Final linear layer
        x = self.lin(x)
        
        return x.flatten()

    def predict(self, data):
        return np.argmax(self(data))

# evaluation of the model
def evaluate(model, val_loader, bs: BosonSampler = None):
    if model.embedding_size:
        outputs = []
        for step, batch in enumerate(tqdm(val_loader)):
            # embedding in the BS
            images, labs = batch
            images = images.squeeze(0).squeeze(0)
            t_s = time.time()
            embs = bs.embed(images,1000)
            outputs.append(model.validation_step(batch, emb=embs.unsqueeze(0)))
    else:
        outputs = [model.validation_step(batch) for batch in val_loader]
    return(model.validation_epoch_end(outputs))