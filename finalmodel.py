import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import copy
import os
import numpy as np
from copy import deepcopy
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
from tqdm import tqdm

df = pd.read_pickle('tmall.pickle')
df.head()

with open('topic_items_to_emb-tmall.pkl','rb') as fp:
    embedding_map = pickle.load(fp)

with open('training_negative.pkl','rb') as fp:
    training_negative = pickle.load(fp)
    
x = []
y = []
all_zero = [0 for _ in range(256)]
for i in tqdm(range(int(len(df)*0.1))):
    row = df.iloc[i]
    past_id = row['past_topic']
    gt = row['future_topic']
    leaves = []
    for j in range(len(past_id)):
        key = tuple((past_id[j],tuple(row['past_leaf'][j])))
        leaves += list(embedding_map[key])
    
    for _ in range(10 - len(past_id)):
        leaves += all_zero
        
    temp = [72 for _ in range(10-len(past_id))]
    xs = []
    xs += past_id
    xs += temp
    xs += leaves
    candidate = list(copy(training_negative[i]))
    candidate += past_id
    for each in set(candidate):
        temp = copy(xs)
        temp.insert(11,each)
        x.append(temp)
        if each in gt:
            y.append(1)
        else:
            y.append(0)
        
x = torch.tensor(x,dtype=torch.float)
y = torch.tensor(y,dtype=torch.float)
x = x.to('cuda:0')
y = y.to('cuda:0')

config = {
    'embedding_dim': 64,
    'num_cate': 72,
    'vae' : 256,
    'dim1' : 1024,
    'dim2' : 256
}

class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        super(user_preference_estimator, self).__init__()
        
        self.embedding_dim = config['embedding_dim']
        self.num_cateogry = config['num_cate'] + 1
        self.vae = config['vae']
        self.fc1_in_dim = config['embedding_dim']  * 10 * 2 + config['embedding_dim']
        self.fc2_in_dim = config['dim1']
        self.fc2_out_dim = config['dim2']

        
        self.embedding = nn.Embedding(self.num_cateogry,self.embedding_dim)
        self.vaeencoder = torch.nn.Linear(self.vae,self.embedding_dim)
        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)
    
    
    def forward(self, x):
        #First ten entries are categories
        category = Variable(x[:, 0:11], requires_grad=False)
        leaf = Variable(x[:,11:], requires_grad=False)
        leaf = leaf.view(-1,10,self.vae)
        out = self.embedding(category.long())
        out = out.view(-1,self.embedding_dim*11)
        out2 = self.vaeencoder(leaf)
        out2 = out2.view(-1,self.embedding_dim*10)
        out = torch.cat((out, out2), 1)
        out = self.fc1(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = F.relu(out)

        out = self.linear_out(out)
        return out
        
estimator = user_preference_estimator(config)
estimator.to('cuda:0')

optimizer = torch.optim.Adam(estimator.parameters(), lr = 0.001)
for _ in range(5):
    for i in tqdm(range(len(x))):
        losses_q = []
        y_pred = estimator.forward(x[i:i+1])
        loss_q = F.mse_loss(y_pred, y[i:i+1].view(-1, 1))
        losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)

        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            losses_q.clamp(min=1e-6)
            losses_q.backward()
        optimizer.step()
    torch.save(estimator.state_dict(),"final.pkl")