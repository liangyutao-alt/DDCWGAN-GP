from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import pandas as pd
def multiclassifier(i):
    colon = pd.read_csv("gene_colonSelect/test_data_"+i+".csv", header=None)
    colon_data = colon.iloc[:, :2000].values
    colon_label = colon.iloc[:, 2000:].values

    # sc = StandardScaler()
    # sc.fit(colon_data)
    #
    # colon_data = sc.transform(colon_data)

    train_data0 = pd.read_csv('createData/CWGANGP_create_colondata0_'+i+'.txt', header=None)
    train_data1 = pd.read_csv('createData/CWGANGP_create_colondata1_'+i+'.txt', header=None)
    train_data0['label'] = 0
    train_data1['label'] = 1
    data = pd.concat([train_data0,train_data1],ignore_index=True)
    label = data.iloc[:, 2000:].values
    data = data.iloc[:, :2000].values

    # data = sc.transform(data)

    sc=StandardScaler()
    sc.fit(data)
    data=sc.transform(data)
    colon_data=sc.transform(colon_data)


##################################################################

    # KNN
    knn=KNeighborsClassifier()
    knn.fit(data,label)
    knnacc=knn.score(colon_data,colon_label)

    # DT
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(data, label)
    treeacc = tree.score(colon_data, colon_label)

    # NB
    nb=GaussianNB()
    nb.fit(data,label)
    nbacc=nb.score(colon_data,colon_label)

    # RF
    rf=RandomForestClassifier()
    rf.fit(data,label)
    rfacc=rf.score(colon_data,colon_label)


    # MLP
    # mlp=MLPClassifier()
    # mlp.fit(data,label)
    # mlpacc=mlp.score(colon_data,colon_label)

    return knnacc,treeacc,nbacc,rfacc

if __name__=="__main__":
    knnacc,dtacc,nbacc,rfacc,mlpacc=multiclassifier("3")
    print("knn:{}  dt:{}   nb:{}  rf:{} mlp:{}".format(knnacc,dtacc,nbacc,rfacc,mlpacc))