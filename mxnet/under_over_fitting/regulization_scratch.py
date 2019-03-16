from mxnet import ndarray as nd
import random


# 还是跟以前一样，在训练的过程中我们需要根据batch_size来每次load这些数据出来，送入模型训练
# 所以这里我们定义一个data迭代器
def data_iter(num_samples):
    idx = list(range(num_samples))
    random.shuffle(idx)

    for i in range(0, num_samples, batch_size):
        j = nd.array(idx[i: min(i + batch_size, num_samples)])
        yield X.take(j), y.take(j)


def get_params():
    learn_w = nd.random_normal(shape=(input_dim, 1)) * 0.1
    learn_b = nd.zeros(1, 1)

    for param in (learn_w, learn_b):
        param.attach_grad()
    return (learn_w, learn_b)


if __name__ == '__main__':
    num_train = 20
    num_test = 100
    # 200维的特征
    input_dim = 200

    batch_size = 1

    true_bias = 0.05
    true_w = nd.ones((input_dim, 1)) * 0.01

    # 生成训练数据和测试数据

    X = nd.random_normal(0, 0.01, shape=(num_train + num_test, input_dim))
    y = nd.dot(X, true_w) + true_bias
    y += .01 * nd.random_normal(shape=y.shape)

    X_train, X_test = X[:num_train, :], X[num_train:, :]
    y_train, y_test = y[:num_test, :], y[num_test:, :]
