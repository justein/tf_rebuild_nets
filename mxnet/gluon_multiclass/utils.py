from mxnet import ndarray as nd
#我们知道max，假如说我有两个数，a和b，并且a>b，如果取max，那么就直接取a，没有第二种可能
#但有的时候我不想这样，因为这样会造成分值小的那个饥饿。所以我希望分值大的那一项经常取到，分值小的那一项也偶尔可以取到，
#那么我用softmax就可以了 现在还是a和b，a>b，如果我们取按照softmax来计算取a和b的概率,
#那a的softmax值大于b的，所以a会经常取到，而b也会偶尔取到，概率跟它们本来的大小有关。
#所以说不是max，而是 Soft max
#也就是说，是该元素的指数，与所有元素指数和的比值
def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis=1, keepdims=True)
    return exp / partition

# 梯度下降，原地操作
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad