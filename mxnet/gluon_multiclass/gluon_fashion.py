from mxnet import gluon as gl
from mxnet import ndarray as nd
from utils import *
from mxnet import autograd as ag

def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')

#def show_imgs():
# todo
#X是一个3D的东西，需要reshape成一个2D的矩阵，-1代表让系统自己infer这个维度是多少，其实这里就是batch_size
# 也就是网络一次读取多少张图片进行训练
# reshape读取batch_size张图片，将图片拉成一条（28*28）784*1的向量，然后乘以权重，加上偏置项，送入softmax
# 即可得到属于每一类的概率是多少,因为X乘以W得到的不是一个概率，如果normallize成一个概率，就是softmax的功能

def net(X):
     return softmax(nd.dot(X.reshape(-1, num_input), W) + b)

# 交叉熵损失函数，这里我们需要定义一个预测结果为概率值的损失函数，它将两个概率分布的负交叉熵作为目标值
# 最小化这个值，相当于最大化这两个概率的相似度
def cross_entropy(yhat, y):
    return -nd.pick(nd.log(yhat), y)
# 准确率，取预测结果中概率最大的一个，看它是否等于label真实值，一次batch中取一个平均
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iter, net):
    acc = 0
    for data, label in data_iter:
        output = net(data)
        acc += accuracy(output, label)
    return  acc / len(data_iter)

if __name__ == '__main__':
    mnist_train = gl.data.vision.FashionMNIST(train=True, transform=transform)
    mnist_test = gl.data.vision.FashionMNIST(train=False, transform=transform)

    data, label = mnist_train[0]
    print('example data : %s , label : %s ' % (data.shape, label))

    batch_size = 256
    train_data = gl.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gl.data.DataLoader(mnist_test, batch_size, shuffle=False)
    # 初始化模型参数
    num_input = 28*28
    num_output = 10
    # 最后的结果是输出该样本属于10类中哪一个类别的概率，所以这里需要的权重是一个 28*28 * 10 的矩阵
    W = nd.random_normal(shape=(num_input, num_output))
    b= nd.random_normal(shape=num_output)

    params = [W, b]

    # 为参数开一个梯度
    for param in params:
        param.attach_grad()
    # 开始训练
    learning_rate = 0.1
    epoch = 10

    for e in range(epoch):
        training_loss = 0.
        train_acc = 0.

        for data, label in train_data:
            with ag.record():
                output = net(data)
                loss = cross_entropy(output, label)
            # 回传loss更新梯度
            loss.backward()
            # 将梯度进行平均化，这样消除对batch size的敏感
            SGD(params, learning_rate / batch_size)
            training_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)

        test_acc = evaluate_accuracy(test_data, net)
        print('Epoch %d. Loss: %f , Train acc: %f , Test acc: %f' % (e,
              training_loss/len(train_data), train_acc / len(train_data), test_acc))
