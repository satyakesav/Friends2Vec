import pandas as pd
import numpy as np
from gmf import GMF
from mlp import MLP
from neumf import NeuMF
from neumf_social import NeuMF_Social
import torch.optim as optim
import math
import torch
from torch.utils import data

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ratings, social_embeddings_dict):
        'Initialization'
        self.ratings  = ratings
        self.social_embeddings = social_embeddings_dict
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ratings)

    def __getitem__(self, index):
        'Generates one sample of data'
        # # Load data and get label
        user_item_pair = self.ratings[index][0:2].astype('float')
        user_social = self.social_embeddings[user_item_pair[0]].astype('float')
        #user_social = np.zeros(64).astype('float')
        user_item_pair_social = np.concatenate((user_item_pair, user_social), axis=None)
        X = user_item_pair_social
        y = self.ratings[index][2:3].astype('float')
        return X, y
    
def epoch_run(model, generator, opt, criterion,mode="train", social=False):
    running_loss = 0
    if(mode == "train"):
        model.train()
    else:
        model.eval()
    for local_batch, local_labels in generator:
        local_batch_social  = torch.tensor(local_batch[:, 2:]).type(torch.float).to(device)
        local_batch  = torch.tensor(local_batch[:, 0:2]).type(torch.long).to(device)
        local_labels = local_labels.type(torch.float).to(device)
        #print(local_batch.size(), local_batch_social.size())
        y_preds = model(local_batch[:,0], local_batch[:,1], local_batch_social)
        loss = criterion(y_preds, local_labels)
        running_loss += (loss.item()*local_labels.size()[0])
        if(mode == "train"):
            opt.zero_grad()
            loss.backward()
            opt.step()
    avg_loss = running_loss * 1.0 / (len(generator.dataset))
    return avg_loss

def predict(model, generator):
    model.eval()
    y_preds_all = torch.Tensor().to(device) 
    y_labels_all = torch.Tensor().to(device) 
    for local_batch, local_labels in generator:
        local_batch_social  = torch.tensor(local_batch[:, 2:]).type(torch.float).to(device)
        local_batch  = torch.tensor(local_batch).type(torch.long).to(device)
        local_labels = local_labels.type(torch.float).to(device)
        with torch.no_grad():
            y_preds = model(local_batch[:,0], local_batch[:,1], local_batch_social)
        y_preds_all = torch.cat((y_preds_all,y_preds))
        y_labels_all = torch.cat((y_labels_all,local_labels))
    return y_preds_all, y_labels_all

def evaluate(model, generator):
    y_preds_all, y_labels_all = predict(model, generator)  
    y_preds = list(y_preds_all.view(1, y_preds_all.size()[0]).to("cpu").numpy()[0])
    y_actuals = list(y_labels_all.view(1, y_labels_all.size()[0]).to("cpu").numpy()[0])
    #print(type(y_preds), type(y_actuals))
    tmse = sum([(a-b) * (a-b) for a,b in zip(y_preds, y_actuals)])
    rmse = math.sqrt((1.0*tmse)/len(y_preds))
    return rmse

def train_gmf():
    model = GMF(gmf_config).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    for epoch in range(gmf_config['num_epoch']):
        print("running epoch ", epoch)
        train_mse = epoch_run(model, train_generator, opt, criterion, "train")
        val_mse = epoch_run(model, val_generator, opt, criterion,"val")
        print("train mse loss => ", train_mse, "val mse loss => ", val_mse)
    return model

def train_mlp():
    model = MLP(mlp_config).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    for epoch in range(mlp_config['num_epoch']):
        print("running epoch ", epoch)
        train_mse = epoch_run(model, train_generator, opt, criterion,"train")
        val_mse = epoch_run(model, val_generator, opt,criterion, "val")
        print("train mse loss => ", train_mse, "val mse loss => ", val_mse)
    return model

def train_neumf():
    model = NeuMF(neumf_config).to(device)
#     if config['pretrain']:  #TODO:: Manoj
#         model.load_pretrain_weights()
    opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    for epoch in range(neumf_config['num_epoch']):
        print("running epoch ", epoch)
        train_mse = epoch_run(model, train_generator, opt,criterion, "train")
        val_mse = epoch_run(model, val_generator, opt,criterion, "val")
        print("train mse loss => ", train_mse, "val mse loss => ", val_mse)
    return model

def train_neumf_social():
    model = NeuMF_Social(neumf_social_config).to(device)
#     if config['pretrain']:  #TODO:: Manoj
#         model.load_pretrain_weights()
    opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    for epoch in range(neumf_social_config['num_epoch']):
        print("running epoch ", epoch)
        train_mse = epoch_run(model, train_generator, opt,criterion, "train")
        val_mse = epoch_run(model, val_generator, opt,criterion, "val")
        print("train mse loss => ", train_mse, "val mse loss => ", val_mse)
    return model

