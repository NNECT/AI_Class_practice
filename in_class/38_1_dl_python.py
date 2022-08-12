import numpy as np
from matplotlib import pyplot as plt


def cost(x, y, w):
    hx = x * w
    return ((hx - y) ** 2).mean()


def gradient_descent(x, y, w):
    hx = x * w
    return ((hx - y) * x).mean()    # (hx - y) ** 2 의 미분
                                    # 2(hx-y) * x   -> 2는 생략


def show_cost(x, y):
    for i in range(-30, 50):
        w = i / 10
        c = cost(x, y, w)
        # print(w, c)
        # plt.plot(w, c, 'ro')
        plt.scatter(w, c)
    plt.show()


def show_gradient_descent(x, y, w, count=10, rate=0.1):
    for i in range(count):
        c = cost(x, y, w)
        g = gradient_descent(x, y, w)
        w -= g * rate
        print(i, w)
        plt.plot(i, c, 'ro')
        plt.plot(i, w, 'bo')
        # plt.scatter(i, w)
    plt.show()


if __name__ == "__main__":
    #  y = ax + b
    # hx = wx + b   -> y = hx가 되도록!
    x = np.arange(1, 4)
    y = np.arange(1, 4)

    # show_cost(x, y)
    show_gradient_descent(x, y, w=5, rate=0.21)
