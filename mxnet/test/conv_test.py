import numpy as np

if __name__ == '__main__':
    f = np.array([1,1,1])
    g = np.array([2,3,2,6])

    fg_conv = np.convolve(f, g)

    print(fg_conv)