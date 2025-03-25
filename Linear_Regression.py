import numpy as np
import matplotlib.pyplot as plt
import time

# region Linear_Regression
class Linear_Regression:
    def __init__(self, W, B):
        self.b = B
        self.w = W
        self.lr = 0.01
        self.m = 100

    def creat_random_X(self):
        np.random.seed(0)
        x = np.random.rand(self.m , 1) * 5
        return x


    def creat_random_Y(self, X):
        np.random.seed(0)
        y = np.random.rand(self.m , 1) + 3 * X + 5 * np.random.rand(self.m , 1)
        return y

    def Gradient_descent(self, X, Y):
        predict_y = self.w * X + self.b
        partialW = np.sum(X * (Y - predict_y)) * -1 / self.m
        partialB = np.sum(Y - predict_y) * -1 / self.m
        self.w = self.w - self.lr * partialW
        self.b = self.b - self.lr * partialB
        return self.w, self.b
# endregion

# region Linear_Regression_2
class Linear_Regression_2:
    def __init__(self, w2, w1, b):
        self.b = b
        self.w2 = w2
        self.w1 = w1
        self.lr = 0.0000001
        self.m = 100

    def create_random_x(self):
        np.random.seed(0)
        random_x = np.random.randint(-10, 10, (self.m , 1)) * 5
        return random_x


    def create_random_y(self, x):
        np.random.seed(0)
        random_y = 15 * np.random.randint(-50, 50, (self.m , 1)) + 2 * np.square(x) + 3 * x
        return random_y

    def gradient_descent(self, x, y):
        predict_y = self.w2 * np.square(x) + self.w1 * x + self.b
        partialW2 = np.sum(np.square(x) * (y - predict_y)) * -1 / self.m
        partialW1 = np.sum(x * (y - predict_y)) * -1 / self.m
        partialB = np.sum(y - predict_y) * -1 / self.m
        self.w2 = self.w2 - self.lr * partialW2
        self.w1 = self.w1 - self.lr * partialW1
        self.b = self.b - self.lr * partialB
        return self.w2, self.w1, self.b
# endregion

if __name__ == "__main__":
    L = Linear_Regression_2(w2 = -0.5, w1 = -0.5, b = -0.5)
    x = L.create_random_x()
    y = L.create_random_y(x)
    print(x)
    print(y)
    plt.ion()

    for i in range(500):
        w2, w1, b = L.gradient_descent(x,y)
        print("w2: " + str(w2) + "  " + "w1: " + str(w1) + "  " + "b: " + str(b))

        if i % 1 == 0:
            plt.clf()
            fit_x = []
            fit_y = []
            for i in range(-60, 60, 1):
                fit_x.append(i)
                fit_y.append(w2 * i ** 2 + w1 * i + b)
            plt.xlim(-100,100)
            plt.ylim(-2000, 10000)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("test")
            plt.scatter(x, y)
            # plt.show()

            plt.plot(fit_x, fit_y)
            plt.pause(0.005)
            plt.ioff()
