import sys
from mxnet import gluon as gl
from mxnet import ndarray as nd
from mxnet import autograd as ag


def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')


# 多层神经网络中比较重要的是有个激活函数，让模型具备拟合非线性函数的能力
def reLU(x):
    return nd.maximum(x, 0)


def net(X):
    # 同样，这里让模型自己infershape的第一维，其实就是batch_size，将图片拉成一个向量
    X = X.reshape((-1, input_dim))
    h1 = reLU(nd.dot(X, learn_w1) + learn_b1)
    out = nd.dot(h1, learn_w2) + learn_b2
    return out


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def evaluate_accuracy(test_data, net):
    acc = 0
    for data, label in test_data:
        out = net(data)
        acc += accuracy(out, label)
    return acc / len(test_data)


# 准确率计算
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()


if __name__ == '__main__':
    # 输入参数shape
    input_dim = 28 * 28
    # 输出类别
    output_num = 10
    # 新加入的隐藏层尺寸
    hidden_layer_size = 256
    # batch大小
    batch_size = 256
    # scale
    scale = .01
    # 构造数据集
    mnist_train = gl.data.vision.FashionMNIST(train=True, transform=transform)
    mnist_test = gl.data.vision.FashionMNIST(train=False, transform=transform)

    train_data = gl.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gl.data.DataLoader(mnist_test, batch_size, shuffle=False)

    # 这里使用两层的网络结构，中间加入隐藏层，思考为什么加入隐藏层后模型精度提高？
    # w1的shape，当前层的输出作为后一层的输入
    learn_w1 = nd.random_normal(shape=(input_dim, hidden_layer_size), scale=scale)
    learn_b1 = nd.zeros(shape=hidden_layer_size)

    learn_w2 = nd.random_normal(shape=(hidden_layer_size, output_num), scale=scale)
    learn_b2 = nd.zeros(shape=output_num)

    params = [learn_w1, learn_b1, learn_w2, learn_b2]
    # 梯度占位符
    for param in params:
        param.attach_grad()

    # 分开实现softmax和交叉熵损失会导致数值不稳定，所以这里直接使用gluon提供的实现
    softmax_cross_entropy = gl.loss.SoftmaxCrossEntropyLoss()

    # 开始训练

    learning_rate = .5
    epoch = 10

    for e in range(epoch):
        train_loss = 0.
        train_acc = 0.

        for data, label in train_data:
            with ag.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            # 梯度回传
            loss.backward()
            # 更新param,这里一定要记得除以batch_size,让lr对batch不敏感
            SGD(params, learning_rate / batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)
            # todo

        test_acc = evaluate_accuracy(test_data, net)
        # 通过训练，我们可以发现，收敛速度明显好于前面的逻辑回归，大家可以思考一下这是为什么
        print('Epoch %d. Loss: %f , Train acc: %f , Test acc: %f' % (e,
                                                                     train_loss / len(train_data),
                                                                     train_acc / len(train_data), test_acc))
