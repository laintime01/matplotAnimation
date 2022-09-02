# pip install matplotlib, numpy, scikit-learn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import random


def make_bar_animation():
    print('make a bar animation')
    values = [0] * 50
    for i in range(50):
        values[i] = random.randint(0, 100)
        plt.xlim(0, 50)
        plt.ylim(0, 100)
        plt.bar(list(range(50)), values)
        plt.pause(0.0001)
    plt.show()


def make_big_bar_animation():
    head_tail = [0, 0]
    for i in range(100000):
        if i % 50 ==0:
            head_tail[random.randint(0, 1)] += 1
            plt.bar([0,1], head_tail, color=("blue", "red"))
            plt.pause(0.001)
    plt.show()

def make_mat_animation():
    plt.clf()
    reg = LinearRegression()
    x_values = []
    y_values = []

    for i in range(1000):
        x_values.append(random.randint(0,100))
        y_values.append(random.randint(0,100))

        x = np.array(x_values)
        x = x.reshape(-1, 1)

        y = np.array(y_values)
        y = y.reshape(-1, 1)

        if i % 5 ==0:
            reg.fit(x, y)
            plt.xlim(0,100)
            plt.ylim(0,100)
            # plt.scatter()函数用于生成一个scatter散点图
            plt.scatter(x_values, y_values, color='black')
            plt.plot(list(range(100)), reg.predict(np.array([x for x in range(100)]).reshape(-1, 1)))
            plt.pause(0.001)
    plt.show()


if __name__ == '__main__':
    make_big_bar_animation()