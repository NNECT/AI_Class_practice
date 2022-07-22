import matplotlib.pyplot as plt
import numpy as np


def plot_1():
    plt.plot([1, 2, 3, 4, 9], 'r')
    plt.plot([2, 3, 4, 5, 6], 'g.')
    plt.plot([3, 4, 5, 6, 7], 'bx')
    plt.show()


def plot_2():
    # x = np.arange(-10, 10.1, 0.1)
    x = np.linspace(-10, 10, 1000)  # 양쪽 끝을 포함
    y = x ** 2
    plt.plot(x, y)
    plt.show()


def plot_3():
    x1 = np.linspace(100, 0.001, 1000)
    x2 = np.linspace(-100, -0.001, 1000)
    y1 = np.log(x1)
    y2 = np.log(-x2)
    plt.plot(x1, y1, 'r')
    plt.plot(x1, -y1, 'b')
    plt.plot(x2, y2, 'g')
    plt.plot(x2, -y2, 'y')
    plt.show()


def plot_4():
    x1 = np.linspace(100, 0.001, 1000)
    x2 = np.linspace(-100, -0.001, 1000)
    y1 = np.log(x1)
    y2 = np.log(-x2)
    plt.subplot(1, 2, 1)
    plt.plot(x1, y1, 'r')
    plt.subplot(1, 2, 2)
    plt.plot(x1, -y1, 'b')
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(x2, y2, 'g')
    plt.subplot(1, 2, 2)
    plt.plot(x2, -y2, 'y')
    plt.show()


def plot_5():
    men = [25, 21, 30, 27, 19]
    women = [31, 27, 26, 17, 28]

    column = np.arange(len(men))
    width = 0.4
    plt.bar(column, men, width=width)
    plt.bar(column + width, women, width=width)
    plt.xticks(column + width / 2, ['a', 'b', 'c', 'd', 'e'])

    plt.show()


if __name__ == '__main__':
    plot_5()
