# 38_1_dl_python.py
import matplotlib.pyplot as plt


def cost(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        c += (hx - y[i]) ** 2

    return c / len(x)


def gradeint_descent(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        # c += (hx - y[i]) ** 2
        # c += 2 * (hx - y[i]) ** (2 - 1) * (hx - y[i])미분
        # c += 2 * (hx - y[i]) * (w * x[i] - y[i])미분
        # c += 2 * (hx - y[i]) * (x[i] - 0)
        c += (hx - y[i]) * x[i]

    return c / len(x)


def show_cost():
    # y = ax + b
    # y =  x
    #     1    0
    # hx= wx + b
    x = [1, 2, 3]
    y = [1, 2, 3]

    print(cost(x, y, w=0))
    print(cost(x, y, w=1))
    print(cost(x, y, w=2))
    print()

    for i in range(-30, 50):
        w = i / 10
        c = cost(x, y, w)

        print(w, c)
        plt.plot(w, c, 'ro')

    plt.show()


def show_gradeint_descent():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = -5
    for i in range(10):
        g = gradeint_descent(x, y, w)
        w -= 0.1 * g

        print(i, w)


# show_cost()
show_gradeint_descent()

# 미분: 기울기, 순간 변화량
#      x가 1만큼 변할 때 y가 변하는 정도

# y = 3                 3=1, 3=2, 3=3
# y = x                 1=1, 2=2, 3=3
# y = 2x                2=1, 4=2, 6=3
# y = (x + 1)           2=1, 3=2, 4=3
# y = xz

# y = x ^ 2             1=1, 4=2, 9=3
#                       2 * x ^ (2 - 1) * x미분 = 2x
# y = (x + 1) ^ 2       2 * (x + 1) ^ (2 - 1) * (x+1)미분 = 2(x + 1)



