from mxnet import ndarray as nd
from mxnet import autograd as ag
import random


# 批量获取数据
def load_data_iter():
    # 索引
    idx = list(range(sample_num))
    # 乱序
    random.shuffle(idx)

    for i in range(0, sample_num, batch_size):
        j = nd.array(idx[i: min(i + batch_size, sample_num)])
        yield nd.take(X, j), nd.take(y, j)


# 定义模型，这里就是一个简单的线性回归
def net(input_data):
    return nd.dot(input_data, learn_w) + learn_b


# 损失函数
def square_loss(yhat, real_y):
    # 防止boradcast
    return (yhat - real_y.reshape(shape=yhat.shape)) ** 2


# 梯度下降，原地操作
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


if __name__ == '__main__':
    # 走多少轮训练集
    epoch = 10
    # 学习率
    learning_rate = .001
    # 样本数量
    sample_num = 1000
    # 样本维度
    input_dim = 2
    # 跑批数目
    batch_size = 10
    # 生成样本数据
    X = nd.random_normal(0, 1, shape=(sample_num, input_dim))
    # 真实权重
    true_w = [2, -3.4]
    # 真实偏置
    true_b = 4.5
    # 生成样本数据
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    # 给数据加噪声
    y += .01 * nd.random_normal(shape=y.shape)
    # 测试前10条
    # print(X[:10],y[:10])
    # 测试批量load数据
    # for data, label in load_data_iter():
    #   print(data, label)
    #    break

    # 权重和偏置参数初始化
    learn_w = nd.random_normal(shape=(input_dim, 1))
    learn_b = nd.zeros((1,))
    params = [learn_w, learn_b]

    # 权重参数梯度占位符
    for param in params:
        param.attach_grad()

    # 正式开始训练
    for e in range(epoch):
        total_loss = 0
        for data, label in load_data_iter():
            with ag.record():
                # 预测值
                out = net(data)
                loss = square_loss(out, label)
                # 回传loss
            loss.backward()
            # 按照梯度更新权重参数
            SGD(params, learning_rate)
        total_loss += nd.sum(loss).asscalar()
        print('Epoch %d , average loss is %f' % (e, total_loss / sample_num))
        # print(learn_w )
