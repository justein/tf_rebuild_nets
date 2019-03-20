from mxnet import ndarray as nd
from mxnet import gluon as gl
import matplotlib.pyplot as plt
from mxnet import autograd as ag

# 定义平方误差函数
# 测试函数
def test(net, X, y):
    return square_loss(net(X), y).mean().asscalar()

def train(X_train, X_test, y_train, y_test):
    # 线性回归模型
    net = gl.nn.Sequential()
    with net.name_scope():
        net.add(gl.nn.Dense(1))
    net.initialize()

    # 超参数
    learning_rate = 0.01
    batch_size = 10
    epoch = 100

    # 构造训练数据
    data_train = gl.data.ArrayDataset(X_train, y_train)
    # 加载训练数据
    data_iter_train = gl.data.DataLoader(data_train, batch_size, shuffle=True)
    # 构造训练器
    trainer = gl.Trainer(net.collect_params(), 'sgd',{'learning_rate':learning_rate})
    # 保存train loss 和test loss用于画图
    train_loss = []
    test_loss = []

    for e in range(epoch):
        for data, label in data_iter_train:
            with ag.record():
                out_put = net(data)
                loss = square_loss(out_put, label)
            loss.backward()
            trainer.step(batch_size)
        # fill the loss array
        train_loss.append(square_loss(net(X_train), y_train).mean().asscalar())
        test_loss.append(square_loss(net(X_test), y_test).mean().asscalar())

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.show()
    print('learning weight ',net[0].weight.data(), 'learning bias: ',net[0].bias.data())
    return('learning weight ',net[0].weight.data(), 'learning bias: ',net[0].bias.data())

if __name__ == '__main__':
    # 真实值
    train_num = 100
    test_num = 100
    true_w = [1.2 ,-3.4, 5.6]
    true_b = 5.0
    square_loss = gl.loss.L2Loss()
    x = nd.random_normal(shape=(train_num+test_num, 1))
    X = nd.concat(x, nd.power(x, 2),nd.power(x, 3))
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_w[2] * X[:,2] + true_b
    y += 0.1 * nd.random_normal(shape=y.shape)

    y_train, y_test = y[:train_num], y[train_num:]

    # 检查一下我们生成的数据集
    # print(x[:5] , X[:5], y[:5])

    #下面就是训练的过程，先使用与数据生成函数同样阶数的三阶多项式来拟合
    # train(X[:train_num, :], X[train_num:, :], y[:train_num], y[train_num:])
    '''
    learning weight  
[[ 1.2132103 -3.2849457  5.758924 ]]
<NDArray 1x3 @cpu(0)> learning bias:  
[5.020751]
<NDArray 1 @cpu(0)>
    '''
    # 下面我们使用线性拟合
    # 当使用线性函数就行拟合的时候，表示只用了训练数据中的一个维度，也就是一个特征，这样学习出来的模型
    # 由于特征太少，训练误差会很高，所以很难在实际中使用，也就说会发生欠拟合
    train(x[:train_num, :], x[train_num:, :], y[:train_num], y[train_num:])
    '''
    learning weight  
[[23.101524]]
<NDArray 1x1 @cpu(0)> learning bias:  
[-0.30870497]
<NDArray 1 @cpu(0)>
    '''
