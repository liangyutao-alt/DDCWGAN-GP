import pandas as pd
import numpy as np
data = pd.read_csv("CWGANGP_Colon_test_acc.txt",header=None)
data = data.values
mean = np.mean(data,axis=0)
std = np.std(data,axis=0)
print("----------------mlp----------------knn--------------dt---------------nb---------rf------")
print(data)
print('mean',mean)
print('std',std)
# print("-----------------------------------------------------\n\n\n")
# data = pd.read_csv("LSGAN_Colon_test_acc.txt",header=None)
# data = data.values
# mean = np.mean(data,axis=0)
# std = np.std(data,axis=0)
# print(data)
# print(mean)
# print(std)