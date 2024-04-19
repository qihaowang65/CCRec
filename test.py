import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import copy
import os
import numpy as np
from copy import copy
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
from tqdm import tqdm

df = pd.read_pickle('tmall.pickle')
with open('topic_items_to_emb-tmall.pkl','rb') as fp:
    embedding_map = pickle.load(fp)

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


with open('testing_negative.pkl','rb') as fp:
    testing_negative = pickle.load(fp)

xt = []
yt = []
all_zero = [0 for _ in range(256)]
cand = []
offset = int(len(df)*0.9)
for i in tqdm(range(int(len(df)*0.9),int(0.91*len(df)))):
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
    candidate = list(copy(testing_negative[i-offset]))
    candidate += past_id
    candidate = list(set(candidate))
    cx = []
    for each in candidate:
        temp = copy(xs)
        temp.insert(11,each)
        cx.append(temp)
    xt.append(cx)
    yt.append(gt)
    cand.append(candidate)

def ComputeScore(ret,gt,t,show=True):
    tp = 0
    fp = 0
    fn = 0
    h = 0
    for i in range(len(ret)):
        ht = 0
        for each in ret[i][0:t]:
            if each in gt[i]:
                tp += 1
                ht = 1
            else:
                fp += 1
        for each in gt[i]:
            if each not in ret[i]:
                fn += 1
        h += ht
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f = 2*p*r/(p+r)
    if show:
        print("HR@" + str(t) + "       :" + str(h/len(ret)))
        print("Precision@" + str(t) + ":" + str(p))
        print("Recall@" + str(t) + "   :" + str(r))
        print("F1@" + str(t) + "       :" + str(f))
        print("*************************************************************")
    return h/len(ret)

def TestModel(m,x,gt,cand):
    ret = []
    for i in tqdm(range(len(x))):
        xc = x[i]
        xc = torch.tensor(xc,dtype=torch.float)
        out = m(xc)
        out = out.detach().numpy().flatten()
        sorting = np.argsort(-1 * out)
        temp = []
        for each in sorting:
            temp.append(cand[i][each])
        ret.append(temp)
    for t in [1,3,5,7,10]:
        ComputeScore(ret,gt,t)

model = user_preference_estimator(config)
model.load_state_dict(torch.load("final.pkl"))
TestModel(model,xt,yt,cand)