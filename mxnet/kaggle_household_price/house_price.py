from mxnet import autograd, ndarray, gluon, init
import pandas as pd
import numpy as np
import d2lzh as d2l


# 定义网络
def net():
    # 线性函数进行拟合
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    net.initialize()
    return net


# 定义评价函数，使用对数均方根误差
def log_rmse(net, features, labels):
    # 将小于1的值设成1，使得取对数时数值更稳定
    clipped_preds = ndarray.clip(net(features), 1, float('inf'))
    rmse = ndarray.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()


# 定义训练函数
def train(net, train_features, train_labels, test_features, test_labels,
          epochs, lr, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(),
                            'adam', {'learning_rate': lr, 'wd': weight_decay})

    for e in range(epochs):
        for X, y in train_iter:
            with autograd.record():
                lo = loss(net(X), y)
            lo.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = ndarray.concat(X_train, X_part, dim=0)
            y_train = ndarray.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        nets = net()
        train_ls, valid_ls = train(nets, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f'
              % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

if __name__ == '__main__':
    # 使用pandas读入数据
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    # 统计数据shape
    # print(train_data.shape)
    # 观察数据
    df = pd.DataFrame(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
    print(df)
    # 训练集和测试集的所有特征（去掉Id列），训练集中拿掉最后的价格列
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    # 对数据集中的数值类型数据进行标准化处理,高斯标准化
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: ((x - x.mean()) / (x.std()))
    )

    # 填充为na的值
    all_features = all_features.fillna(0)
    # 检查一下数值类型特征标准化以后的训练集，其实这里我们主要是为了跟后面的非数值型特征值标准化以后的shape
    # 进行比较，这里我们只是去掉了id列和房屋售价列
    print(all_features.shape)
    '''(2919, 79)'''
    # 对非数值型的特征进行标准化,这里将 类别变量向量化处理
    # dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
    all_features = pd.get_dummies(all_features, dummy_na=True)
    # 处理之后的数据集shape
    print(all_features.shape)

    # 最后，通过values属性得到NumPy格式的数据，并转成NDArray方便后面的训练。
    # 训练集数目
    num_train = train_data.shape[0]
    # 标准化处理后的训练集和测试集数据
    train_features = ndarray.array(all_features[:num_train].values)
    test_features = ndarray.array(all_features[num_train:].values)

    train_labels = ndarray.array(train_data.SalePrice.values).reshape((-1, 1))

    loss = gluon.loss.L2Loss()

    k, num_epochs, lr, weight_decay, batch_size = 5, 500, 5, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                              weight_decay, batch_size)
    print('%d-fold validation: avg train rmse %f, avg valid rmse %f'
          % (k, train_l, valid_l))