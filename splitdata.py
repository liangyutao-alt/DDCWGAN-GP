import pandas as pd
import numpy as np
import random
def dataTrainTest(m):
    k=6
    weight=0.5
    colon = pd.read_csv("../../../gene_data/data1/Colon.arff", header=None)

    label = colon.iloc[:, 2000:].values
    data = colon.iloc[:, :2000].values

    label[label == "Tumor"] = 1
    label[label == "Normal"] = 0

    def loOP(train,n,extent=2):
        from PyNomaly  import loop
        prob=loop.LocalOutlierProbability(train,extent=extent,n_neighbors=n).fit()
        scores=prob.local_outlier_probabilities.reshape(train.shape[0],1)
        return  scores

    def probNN(x_train,y_train,n):
        from  sklearn.neighbors import  NearestNeighbors
        nbrs=NearestNeighbors(n_neighbors=n+1,algorithm='auto').fit(x_train)
        _,idx=nbrs.kneighbors(x_train)
        idx=idx[:,1:]
        scores=[]
        for i in range(len(y_train)):
            if y_train[i]==0:
                scores.append(np.sum(y_train[idx[i]]==0)/n)
            if y_train[i]==1:
                scores.append(np.sum(y_train[idx[i]]==1)/n)
        scores=np.array(scores,dtype='float32').reshape(x_train.shape[0],1)
        return  scores


    def merge( prob_nn,prob_lof, weight):
        dat_info = (weight * prob_nn + (1 - weight) * (1 - prob_lof))

        return dat_info

    s1=probNN(data,label,k)
    s1=np.squeeze(s1)
    # print(s1)
    index=np.argsort(s1)
    # 从小到大排序
    # print(index)


    s2=loOP(data,k)
    s2=np.squeeze(s2)
    # print(s2)
    index2=np.argsort(-s2)
    # 从大到小排序
    # print(index2)
    # print(s2[index2])

    s3=merge(s1,s2,weight)
    s3=np.squeeze(s3)
    index3=np.argsort(s3)
    # print(index3)

    # import matplotlib.pyplot as plt
    # import matplotlib.colors
    #
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    #
    # X_pca = pca.fit(data).transform(data)

    # plt.figure(figsize=(6,6))
    # color = ["blue", "red"]
    # target_names = ["0", "1"]
    # lw = 3
    # print(label[:,0]==1)
    # for color, i, target_name in zip(color, [0, 1], target_names):
    #     plt.scatter(X_pca[label[:,0] == i, 0], X_pca[label[:,0] == i, 1], color=color, alpha=.4, lw=lw,
    #                 label=target_name)
    #     plt.legend(loc='best', shadow=False, scatterpoints=1)
    #     plt.title('PCA')
    # plt.savefig("5all_Colon.png", dpi=750, bbox_inches='tight')
    # plt.show()

    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=s1)
    # plt.colorbar()
    # plt.savefig("5knn_Colon.png", dpi=750, bbox_inches='tight')
    # plt.show()
    #
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=s2)
    # plt.colorbar()
    # plt.savefig("5lop_Colon.png", dpi=750, bbox_inches='tight')
    # plt.show()

    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=s3)
    # plt.colorbar()
    # plt.savefig("5select_Colon.png", dpi=750, bbox_inches='tight')
    # plt.show()
    #
    #
    # print ("finish")


    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    Index0=[]
    Index1=[]

    test_splitrate=0.2
    ind=index3

    # 按类别数据分类
    for i in range(len(ind)):
        if(label[ind[i]]==0):
            Index0.append(ind[i])
        if (label[ind[i]] == 1):
            Index1.append(ind[i])

    # 标签是0，划分数据集
    test_index0=Index0[-int(test_splitrate*2*len(Index0)):].copy()
    test_index0=np.random.choice(test_index0,int(0.2*len(Index0)),replace=False)
    ind0 = []
    for j in range(len(test_index0)):
        for k in range(len(Index0)):
            if test_index0[j] == Index0[k]:
                ind0.append(k)
                break

    ind0 = np.array(ind0)
    train_index0=np.delete(Index0,ind0)
    # print(index0)
    # print(test_index0)
    # print(train_index0)

    # 标签是1，划分数据集
    test_index1=Index1[-int(test_splitrate*2*len(Index1)):].copy()
    test_index1=np.random.choice(test_index1,int(0.2*len(Index1)),replace=False)
    ind1 = []
    for j in range(len(test_index1)):
        for k in range(len(Index1)):
            if test_index1[j] == Index1[k]:
                ind1.append(k)
                break
    ind1= np.array(ind1)
    train_index1=np.delete(Index1,ind1)
    # print(index1)
    # print(test_index1)
    # print(train_index1)

    test_data0=data[test_index0]
    test_data1=data[test_index1]
    train_data0=data[train_index0]
    train_data1=data[train_index1]
    # print(len(test_data0))
    # print(len(test_data1))
    # print(len(train_data0))
    # print(len(train_data1))
    test_data0 = np.append(test_data0, np.array([[0] for i in range(len(test_data0))]), axis=1)
    test_data1 = np.append(test_data1, np.array([[1] for i in range(len(test_data1))]), axis=1)
    train_data0 = np.append(train_data0, np.array([[0] for i in range(len(train_data0))]), axis=1)
    train_data1 = np.append(train_data1, np.array([[1] for i in range(len(train_data1))]), axis=1)


    def saveDataset(dir, name, dt):
        dat = dt[0]
        for i in range(len(dt) - 1):
            dat = np.append(dat, dt[i + 1], axis=0)
        dt = pd.DataFrame(dat)
        path = dir + name+m + ".csv"
        dt.to_csv(path, header=None, index=False)


    save_dir = "gene_colonSelect/"
    saveDataset(save_dir, "test_data_", [test_data0, test_data1])
    saveDataset(save_dir, "train_data_", [train_data0, train_data1])

if __name__ == '__main__':
        dataTrainTest("0")