from mxnet import gluon as gl
from mxnet import ndarray as nd


def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')

#def show_imgs():
# todo

if __name__ == '__main__':
    mnist_train = gl.data.vision.FashionMNIST(train=True, transform=transform)
    mnist_test = gl.data.vision.FashionMNIST(train=False, transform=transform)

    data, label = mnist_train[0]
    print('example data : %s , label : %s ' % (data.shape, label))
