from mxnet import gluon as gl
from mxnet import autograd as ag
from mxnet import ndarray as nd


def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')


def accuracy(out, label):
    return nd.mean(out.argmax(axis=1) == label).asscalar()


def evaluate_accuracy(dataset, net):
    acc = 0.
    for data, label in dataset:
        out = net(data)
        acc += accuracy(out, label)
    return acc / len(dataset)


if __name__ == '__main__':
    # 定义一个线性网络
    net = gl.nn.Sequential()
    # 因为输入为28*28*1 的3d图片，所以首层需要拉平一波
    with net.name_scope():
        net.add(gl.nn.Flatten())
        # 添加一个有256个节点的DenseLayer
        net.add(gl.nn.Dense(256, activation='relu'))
        # 因为输出为10个类别，所以这里设置节点个数为10
        net.add(gl.nn.Dense(10))
    # 看一波模型的样子
    '''Sequential(
        (0): Flatten
    (1): Dense(None -> 256, Activation(relu))
    (2): Dense(None -> 10, linear)
    )'''
    print(net)
    # 模型初始化
    net.initialize()
    # 超参数们
    batch_size = 256
    epoch = 5
    learning_rate = 0.5

    # load 数据
    mnist_train = gl.data.vision.FashionMNIST(train=True, transform=transform)
    mnist_test = gl.data.vision.FashionMNIST(train=False, transform=transform)

    train_data = gl.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gl.data.DataLoader(mnist_test, batch_size, shuffle=False)

    # 损失函数
    softmax_cross_entropy = gl.loss.SoftmaxCrossEntropyLoss()

    # trainer
    trainer = gl.Trainer(net.collect_params(), 'SGD', {'learning_rate': 0.1})

    for e in range(epoch):
        train_loss = 0.
        train_acc = 0.
        test_acc = 0.

        for data, label in train_data:
            with ag.record():
                out = net(data)
                loss = softmax_cross_entropy(out, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(out, label)
        test_acc = evaluate_accuracy(test_data, net)
        print('Epoch %d. Loss: %f , Train acc: %f , Test acc: %f' % (e,
                                                                 train_loss / len(train_data),
                                                                 train_acc / len(train_data), test_acc))
