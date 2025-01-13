import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from  sklearn.model_selection import  train_test_split
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
# from sklearn.preprocessing import MinMaxScaler
# minmax_scater=MinMaxScaler(feature_range=[-1,1])
import pickle
import  shutil
import os
import  random


import MLPClassify2
import KNNandDTandNBandRF
import splitdata


# 创建文件夹
os.makedirs("DandGLoss",exist_ok=True)
os.makedirs("createData",exist_ok=True)
os.makedirs("Model",exist_ok=True)
os.makedirs("gene_colonSelect",exist_ok=True)
os.makedirs("datasc",exist_ok=True)
os.makedirs("wds",exist_ok=True)



# 将文件夹下的内容全部删除
def del_file(path_data):
    for i in os.listdir(path_data) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i#当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)

del_file("DandGLoss")
del_file("createData")
del_file("Model")
del_file("DandGLoss")
del_file("gene_colonSelect")
del_file("datasc")
del_file("wds")


def read(m):
    splitdata.dataTrainTest(m)
    data=pd.read_csv("gene_colonSelect/train_data_"+m+".csv",header=None)
    label = data.iloc[:, 2000:].values
    data = data.iloc[:, :2000].values

    sc=StandardScaler()
    sc.fit(data)
    data = sc.transform(data)

    yy = [[0], [1]]
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(yy)
    Ytrain_reshape = label.reshape(-1, 1)
    label = encoder.transform(Ytrain_reshape)

    pickle.dump(sc,open("datasc/sc"+m+".pkl",'wb'))
    return data,label,sc



cuda = True if torch.cuda.is_available() else False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dim = 2000
noise_dim = 100
label_dim = 2
condition_dim = label_dim
size = label_dim
G1_dim = 512
G2_dim = 1024
G3_dim = data_dim
D1_dim = 512
D2_dim = 128
D3_dim = 1


# 设置随机种子
# total_seed=35
# random.seed(total_seed)
# np.random.seed(total_seed)
# torch.manual_seed(total_seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(total_seed)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,nn.Linear):
            m.weight.data.normal_(0,0.02)
            m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(noise_dim+label_dim, G1_dim),
            *block(G1_dim, G2_dim),
            nn.Linear(G2_dim, G3_dim),
            nn.Tanh()
        )

        initialize_weights(self)
    def forward(self, z, label):
        input = torch.cat((z,label), -1)
        output = self.model(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            *block(data_dim + label_dim, D1_dim),
            *block(D1_dim, D2_dim),
            *block(D2_dim, D2_dim),
            *block(D2_dim, noise_dim),

        )
        self.fc=nn.Sequential(nn.LeakyReLU(0.2, inplace=False),nn.Linear(noise_dim,D3_dim),)

        initialize_weights(self)
    def forward(self, input, label):
        inputs = torch.cat((input, label), -1)
        latent = self.model(inputs)
        validity = self.fc(latent)
        return latent, validity

def next_batch(train_data, train_target, batch_size, c):
    datas = []
    labels = []
    if c == 0:
        for i in range(len(train_target)):
            if train_target[i][0] == 1 and train_target[i][1] == 0:
                datas.append(train_data[i])
                labels.append(train_target[i])
    elif c == 1:
        for i in range(len(train_target)):
            if train_target[i][0] == 0 and train_target[i][1] == 1:
                datas.append(train_data[i])
                labels.append(train_target[i])
    else:
        datas = train_data
        labels = train_target
    # 打乱数据集
    index = [i for i in range(0, len(datas))]
    np.random.shuffle(index)
    # 建立batch_data与batch_target的空列表
    batch_data = []
    batch_target = []
    # 向空列表加入训练集及标签
    for i in range(0, batch_size):
        batch_data.append(datas[index[i]])
        batch_target.append(labels[index[i]])
    return batch_data, batch_target  # 返回

batch_size = 10
epoch = 3000
learning_rate = 0.0002


from  scipy.stats import  pearsonr
def cal_pcc(z,encode_latent):
    PCC=[]
    z=z.cpu().detach().numpy()
    encode_latent=encode_latent.cpu().detach().numpy()
    pcc=pearsonr(z,encode_latent)[0]
    PCC.append(pcc)
    PCC=torch.from_numpy(np.float32(np.array(PCC))).cuda()

    return PCC

def compute_gradient_penalty(D, real_samples,real_label, fake_samples):
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    mix_E,d_interpolates = D(interpolates,real_label)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1)) ** 2).mean()
    return gradient_penalty,mix_E

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

##############################################MMD
def guassian_kernel(source, target, kernel_mul = 2.0, kernel_num = 5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
        source: 源域数据（n * len(x))
        target: 目标域数据（m * len(y))
        kernel_mul:
        kernel_num: 取不同高斯核的数量
        fix_sigma: 不同高斯核的sigma值
    Return:
        sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0) #将source和target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)

def get_MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
        计算源域数据和目标域数据的MMD距离
        Params:
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_mul:
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
            loss: MMD loss
        '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 将核矩阵分为4份
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss

#############################################

def train(data,label,j):
    DLoss = []
    GLoss = []
    WDS=[]
    MMD=[]

    I_c=0.08
    beta=0
    alpha=1e-5


    def VDB_loss(real_validity, fake_validity,gradient_penalty, pcc_real,pcc_fake, pcc_mix,beta):
        normal_D_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
        # print("real",torch.abs(torch.mean(pcc_real)))
        # print("fake",torch.abs(torch.mean(pcc_real)))
        real_kldiv_loss = torch.abs(torch.mean(pcc_real)) - I_c
        fake_kldiv_loss = torch.abs(torch.mean(pcc_fake)) - I_c
        mix_kldiv_loss  =torch.abs(torch.mean(pcc_mix))-I_c

        # print("real_kldiv_loss", real_kldiv_loss)
        # print("fake_kldiv_loss", fake_kldiv_loss)

        kldiv_loss=real_kldiv_loss+fake_kldiv_loss

        final_loss=normal_D_loss+beta*kldiv_loss
        return  final_loss,kldiv_loss

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=learning_rate)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=learning_rate)

    for e in range(epoch):
        d_epoch_loss = 0
        g_epoch_loss = 0
        w_epoch=0
        mmd_epoch=0

        channel = e % size
        count=len(data)//batch_size
        for i in  range(count):
            batch_data, batch_label = next_batch(data, label, batch_size, channel)
            batch_data = torch.from_numpy(np.float32(np.array(batch_data))).cuda()
            batch_label = torch.from_numpy(np.float32(np.array(batch_label))).cuda()
            batch_noise = torch.randn(size=(batch_size,noise_dim)).cuda()
            disc_iters = 2
            for x in range(0, disc_iters):
                optimizer_D.zero_grad()

                G_z = generator(batch_noise, batch_label)
                E_x, real_validity = discriminator(batch_data, batch_label)
                E_G_z, fake_validity = discriminator(G_z.detach(), batch_label)

                mean_E_x = torch.mean(E_x, dim=0)
                mean_E_G_z = torch.mean(E_G_z, dim=0)
                mean_batch_noise = torch.mean(batch_noise, dim=0)

                gradient_penalty,mix_E= compute_gradient_penalty(discriminator, batch_data.data, batch_label.data, G_z.data)

                mean_mix_E=torch.mean(mix_E,dim=0)
                pcc_real = cal_pcc(mean_batch_noise, mean_E_x)
                pcc_fake = cal_pcc(mean_batch_noise, mean_E_G_z)
                pcc_mix  = cal_pcc(mean_batch_noise,mean_mix_E)

                # print("pcc_mix",pcc_mix)
                d_loss, loss_kldiv = VDB_loss(real_validity, fake_validity, gradient_penalty, pcc_real.data,
                                              pcc_fake.data, pcc_mix.data,beta)

                # print(loss_kldiv)
                beta = max(0.0, beta + alpha * loss_kldiv.detach().item())

                # d_loss.backward(torch.ones_like(d_loss))
                d_loss.backward()
                optimizer_D.step()

                mmd=get_MMD(batch_data.data,G_z.data)

                wds = torch.mean(real_validity) - torch.mean(fake_validity)
                w_epoch += wds.item()
                d_epoch_loss += d_loss.item()
                mmd_epoch += mmd.item()


            optimizer_G.zero_grad()
            G_z = generator(batch_noise, batch_label)
            E_G_z, fake_validity = discriminator(G_z, batch_label)
            E_x,  real_validity =discriminator( batch_data,batch_label)
            # mmd=get_MMD(E_G_z,E_x)
            # print("mmd",mmd)

            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()

            g_epoch_loss += g_loss.item()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [beta :%f][mmd:%f]"
                % (e + 1, epoch, i + 1, count, d_loss.item(), g_loss.item(), beta,mmd.item())
            )

        d_epoch_loss /= (count * disc_iters)
        w_epoch /= (count * disc_iters)
        g_epoch_loss /= count
        mmd_epoch/=(count*disc_iters)

        DLoss.append(d_epoch_loss)
        GLoss.append(g_epoch_loss)
        WDS.append(w_epoch)
        MMD.append(mmd_epoch)

    np.savetxt("DandGLoss/DLoss_" + j + ".txt", DLoss, "%s", delimiter=',')
    np.savetxt("DandGLoss/GLoss_" + j + ".txt", GLoss, "%s", delimiter=',')
    np.savetxt("wds/Wds_" + j + ".txt", WDS, "%s", delimiter=',')
    np.savetxt("MMD/MMD_" + j + ".txt", MMD, "%s", delimiter=',')


    torch.save(generator.state_dict(), "Model/CWGANGP_colon_" + j + ".pth")


def test(create_num, c,i,sc):
    generatorTest = Generator().cuda()
    generatorTest.load_state_dict(torch.load("Model/CWGANGP_colon_" + i + ".pth"))

    sample_noise = torch.from_numpy(np.float32(np.random.uniform(-1, 1, size=(create_num, noise_dim)))).cuda()

    if c == 0:
        condition_d = [[1, 0] for i in range(create_num)]
    elif c == 1:
        condition_d = [[0, 1] for i in range(create_num)]
    condition_d = torch.from_numpy(np.float32(np.array(condition_d))).cuda()
    gen_samples = generatorTest(sample_noise, condition_d).cpu().detach().numpy()
    gen_samples = sc.inverse_transform(gen_samples)

    if c == 1:
        np.savetxt('createData/CWGANGP_create_colondata1_'+i+'.txt', gen_samples, '%s', delimiter=',')
    elif c == 0:
        np.savetxt('createData/CWGANGP_create_colondata0_'+i+'.txt', gen_samples, '%s', delimiter=',')

if __name__=='__main__':
    acc = []

    for i in range(1):
        j = str(i)

        data,label,sc=read(j)

        train(data,label,j)
        test(2000, 0, j,sc)
        test(2000, 1, j,sc)

        mlpacc = MLPClassify2.mlp(j)
        knnacc, dtacc, nbacc, rfacc = KNNandDTandNBandRF.multiclassifier(j)
        print("------mlp--------")
        print(mlpacc)
        print("------knn--------")
        print(knnacc)
        print("--------dt--------")
        print(dtacc)
        print("--------nb-------")
        print(nbacc)
        print("--------rf--------")
        print(rfacc)
        acc.append([mlpacc, knnacc, dtacc, nbacc, rfacc])

    acc = np.array(acc)
    np.savetxt('CWGANGP_Colon_test_acc.txt', acc, "%s", delimiter=',')