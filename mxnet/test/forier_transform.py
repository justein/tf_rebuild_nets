import numpy as np
from pylab import *
if __name__ == '__main__':
    # 时域值
    g1 = np.array([1,1,1,1,1,1,1,1])
    g2 = np.array([1,2,1,2,1,2,1,2])
    # 频域值
    fftg1 = np.fft.rfft(g1)
    fftg2 = np.fft.rfft(g2)

    print(fftg1)
    print(fftg2)

    # 绘制频谱
    figure()
    subplot(121)
    plot(abs(fftg1))
    subplot(122)
    plot(abs(fftg2))
    show()