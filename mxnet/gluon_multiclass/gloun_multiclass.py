from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon as gl

if __name__ == '__main__':

    input_dis = 2
    sample_num = 1000
    true_w = [2, -3.4]
    true_b = 4.5

    X = nd.random_normal(shape=(sample_num, input_dis))
    y = true_w[0]*X[:,0] + true_w[1]*X[:,1] + true_b
    y += .01 * nd.random_normal(shape=y.shape)

    # 数据读取
    batch_size = 10
    dataset = gl.data.ArrayDataset(X, y)
    data_iter = gl.data.DataLoader(dataset, batch_size, shuffle=True)

    # for data, label in data_iter:
    #     print(data, label)

    # 定义模型
    # 当我们手写模型的时候，需要先声明模型参数，然后通过他们来构建模型，gluon定义了很多模型，是内置的。
    # 我们在定义一些简单模型的时候，可以直接使用它们来构建我们的NN，比如线性模型就是使用的DenseLayer
    # 先从最简单的模型开始，构建模型最简单的办法就是使用Sequential将所有层串起来
    #首先定义一个Seq
    net = gl.nn.Sequential()
    #然后加入一个Dense层,它唯一需要指定的就是输出节点的个数，因为我们是线性模型，所以这里是1，
    #这里我们并没有指定输入节点的是多少，这个在后面真正赋值的时候会自动指定
    net.add(gl.nn.Dense(1))
    print(net)
    # 初始化模型参数
    net.initialize()

    #损失函数
    square_loss = gl.loss.L2Loss()
    #优化
    trainer = gl.Trainer(net.collect_params(), 'sgd',{'learning_rate':0.1})
    #训练
    # 这里不再需要调用SGD
    epoch = 5
    for e in range(epoch):
        total_loss = 0
        for data, label in data_iter:
            with ag.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            total_loss += nd.sum(loss).asscalar()
        print('Epoch is %d, avg loss : %f' % (e, total_loss / sample_num))

    # 下面我们就是检验一下模型学习到的参数跟我们的真实参数之间的差别
    print(true_w)
    dense = net[0]
    print(dense.weight.data())
    # 比较偏置项
    print(true_b)
    print(dense.bias.data())