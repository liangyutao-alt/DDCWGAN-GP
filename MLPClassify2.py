import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
import  torch
import  torch.utils.data as Data
import  hiddenlayer as h1

cuda=True if torch.cuda.is_available() else False
print(cuda)
'''
对测试集拟合标准化，在用这个标准化再去转变生成数据
'''

def mlp(i):
    colon = pd.read_csv("gene_colonSelect/test_data_"+i+".csv", header=None)
    colon_data = colon.iloc[:, :2000].values
    colon_label = colon.iloc[:, 2000:].values

    # sc = StandardScaler()

    # sc.fit(colon_data)
    # colon_data = sc.transform(colon_data)

    train_data0 = pd.read_csv('createData/CWGANGP_create_colondata0_'+i+'.txt', header=None)
    train_data1 = pd.read_csv('createData/CWGANGP_create_colondata1_'+i+'.txt', header=None)
    train_data0['label'] = 0
    train_data1['label'] = 1
    data = train_data0.append(train_data1)

    label = data.iloc[:, 2000:].values
    data = data.iloc[:, :2000].values

    # data = sc.transform(data)

    sc = StandardScaler()
    sc.fit(data)
    data = sc.transform(data)
    colon_data = sc.transform(colon_data)

    def next_batch(train_data, train_target, batch_size):
        # 打乱数据集
        index = [i for i in range(0, len(train_target))]
        np.random.shuffle(index)
        # 建立batch_data与batch_target的空列表
        batch_data = []
        batch_target = []
        # 向空列表加入训练集及标签
        for i in range(0, batch_size):
            batch_data.append(train_data[index[i]])
            batch_target.append(train_target[index[i]])
        return batch_data, batch_target  # 返回

    data_dim = 2000
    h1_dim = 256
    h2_dim = 128
    label_dim = 2
    class MLPclassify(nn.Module):
        def __init__(self,data,h1,h2,lab):
            super(MLPclassify,self).__init__()

            self.hidden1=nn.Sequential(
            nn.Linear(data,h1),
            nn.LeakyReLU(),
            # nn.Dropout()
            )

            self.hidden2 = nn.Sequential(
                nn.Linear(h1, h2),
                nn.LeakyReLU(),
                # nn.Dropout(),
            )

            self.class_out = nn.Sequential(
                nn.Linear(h2, lab),
             )

        def forward(self,x):
            fc1=self.hidden1(x)
            fc2=self.hidden2(fc1)
            output=self.class_out(fc2)

            return output

    mlpc=MLPclassify(data_dim,h1_dim,h2_dim,label_dim)
    loss_func=nn.CrossEntropyLoss()
    optimazer=torch.optim.Adam(mlpc.parameters(),lr=0.001)

    if cuda:
        mlpc.cuda()
        loss_func.cuda()

    train_epoch = 400
    batch_size = 1000
    total_size = int(len(data) / batch_size)

    for epoch in range(train_epoch):
        for i in range(total_size):
            x,y=next_batch(data,label,batch_size)
            x = Variable(torch.from_numpy(np.float32(np.array(x)))).cuda()
            y = Variable(torch.from_numpy(np.int64(np.array(y)))).cuda()
            output=mlpc(x)
            y=torch.squeeze(y.long())
            loss=loss_func(output,y)
            optimazer.zero_grad()
            loss.backward()
            optimazer.step()
            print(
                "MLP training :======>[Epoch %d/%d] [Batch %d/%d] [ loss: %f] "
                % (epoch + 1, train_epoch, i + 1, (total_size), loss.item())
            )

    colon_data=Variable(torch.from_numpy(np.float32(np.array(colon_data)))).cuda()
    colon_label=Variable(torch.from_numpy(np.float32(np.array(colon_label)))).cuda()
    testout=mlpc(colon_data)
    pred=testout.data.cpu().numpy()
    gt=colon_label.data.cpu().numpy()
    pt=np.argmax(pred,axis=1).reshape(-1,1)
    acc = np.mean(pt == gt)
    return acc

if __name__ =='__main__':
   print(mlp("3"))