import numpy as np 
import pandas as pd 
df = pd.read_pickle('tmall.pickle')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pickle


class MLE(nn.Module):
    def __init__(self):
        super(MLE, self).__init__()
        self.embedding = nn.Embedding(73,64)
        self.linear1 = nn.Linear(640,320)
        self.linear2 = nn.Linear(320,72)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        out = self.embedding(inputs)
        out = out.view(-1,640)
        out = F.relu(self.linear1(out))
        #out = F.relu(self.linear1(torch.flatten(out)))
        out = self.linear2(out)
        out = self.softmax(out)
        return out
    
    def Probability(self, inputs):
        out = self.embedding(inputs)
        out = out.view(-1,640)
        out = F.relu(self.linear1(out))
        #out = F.relu(self.linear1(torch.flatten(out)))
        out = self.linear2(out)
        return out


x = []
y = []
for i in tqdm(range(int(len(df)*0.9))):
    row = df.iloc[i]
    past_id = row['past_topic']
    gt = row['future_topic']
    temp = [72 for _ in range(10-len(past_id))]
    xs = []
    xs += past_id
    xs += temp
    for each in gt:
        x.append(xs)
        y.append(each)
        
x = torch.tensor(x,dtype=torch.long)
y = torch.tensor(y,dtype=torch.long)

device = 'cuda:0'
x = x.to(device)
y = y.to(device)


losses = []
loss_function = nn.NLLLoss()
mam = MLE()
mam.to(device)
optimizer = optim.SGD(mam.parameters(),lr=0.001)


for epoch in range(10):
    total_loss = 0
    for i in tqdm(range(len(x))):
        mam.zero_grad()
        log_probs = mam(x[i:i+1])
        loss = loss_function(log_probs,y[i:i+1])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    torch.save(mam.state_dict(),"MLE.pkl")

xt = []
yt = []
for i in tqdm(range(int(len(df)*0.9),len(df))):
    row = df.iloc[i]
    past_id = row['past_topic']
    gt = row['future_topic']
    temp = [72 for _ in range(10-len(past_id))]
    xs = []
    xs += past_id
    xs += temp
    xt.append(xs)
    yt.append(gt)
xt = torch.tensor(xt,dtype=torch.long)
x = x.to('cpu')
y = y.to('cpu')
model = MLE()
model.load_state_dict(torch.load("MLE.pkl"))

ret = []
for i in tqdm(range(len(x))):
    out = model.Probability(x[i:i+1]).detach().numpy()[0]
    sorting = np.argsort(-1 * out)
    ret.append(sorting[0:10])

with open('training_negative.pkl','wb') as fp:
    pickle.dump(ret,fp)


ret = []
for i in tqdm(range(len(xt))):
    out = model.Probability(xt[i:i+1]).detach().numpy()[0]
    sorting = np.argsort(-1 * out)
    ret.append(sorting[0:10])

with open('testing_negative.pkl','wb') as fp:
    pickle.dump(ret,fp)
