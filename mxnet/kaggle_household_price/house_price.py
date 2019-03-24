from mxnet import autograd, ndarray, gluon, init
import pandas as pd
import numpy as np

if __name__ == '__main__':

    # 使用pandas读入数据
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    # 统计数据shape
    #print(train_data.shape)
    # 观察数据
    train_data.iloc[0:4, [0,1,2,3,-3,-2,-1]]
