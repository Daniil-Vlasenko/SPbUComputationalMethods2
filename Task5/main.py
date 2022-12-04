import math
import numpy as np
import pandas as pd

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


class VariableDirectionMethod:
    def __init__(self, eps, xLength, yLength, xNods, yNods):
        self.eps = eps
        self.xLength = xLength
        self.yLength = yLength
        self.length = self.xLength # !
        self.xNods = xNods
        self.yNods = yNods
        self.xStep = xLength / xNods
        self.yStep = yLength / yNods
        self.numberOfIterations = 0
        self.step = self.xStep  # !
        self.firstApproximation = None
        self.lastApproximation = None
        self.uGridMatrix = self.uGrid(self.xNods, self.yNods)
        self.fGridMatrix = self.fGrid(self.xNods, self.yNods)
        self.delta1 = 4 / (self.step * self.step) * math.sin(math.pi * self.step / (2 * self.length)) * \
                      math.sin(math.pi * self.step / (2 * self.length))
        self.delta2 = 4 / (self.step * self.step) * math.cos(math.pi * self.step / (2 * self.length)) * \
                      math.cos(math.pi * self.step / (2 * self.length))
        self.tau = 2 / ((self.delta1 * self.delta2) ** (1 / 2))
        self.A = self.tau / (2 * self.step * self.step)
        self.B = self.tau / (self.step * self.step) + 1
        self.C = self.A
        self.informationTable = pd.DataFrame(columns=["rel.error", "sp.rad._k"])

    def uFunction(self, x, y):
        return 2 * x ** 3 * y ** 3

    def fFunction(self, x, y):
        return -12 * x * y * (y ** 2 + x ** 2)

    def terminationMethod(self):
        # return self.numberOfIterations > 100
        return np.max(np.abs(self.lastApproximation - self.uGridMatrix)) / np.max(
            np.abs(self.firstApproximation - self.uGridMatrix)) < self.eps

    def uGrid(self, xNods, yNods):
        return np.array([[self.uFunction(i / xNods, j / yNods) for j in range(yNods + 1)] for i in range(xNods + 1)])

    def fGrid(self, xNods, yNods):
        return np.array([[self.fFunction(i / xNods, j / yNods) for j in range(yNods + 1)] for i in range(xNods + 1)])

    def LuGrid(self, uGridMatrix):
        yNods = len(uGridMatrix) - 1
        xNods = len(uGridMatrix[0]) - 1
        LuResultGrid = np.zeros((yNods + 1, xNods + 1))
        for i in range(1, yNods):
            for j in range(1, xNods):
                LuResultGrid[i][j] = (uGridMatrix[i + 1][j] - uGridMatrix[i][j]) / (self.step ** 2) - \
                                     (uGridMatrix[i][j] - uGridMatrix[i - 1][j]) / (self.step ** 2) + \
                                     (uGridMatrix[i][j + 1] - uGridMatrix[i][j]) / (self.step ** 2) - \
                                     (uGridMatrix[i][j] - uGridMatrix[i][j - 1]) / (self.step ** 2)
        return LuResultGrid

    def A1Grid(self, uGridMatrix):
        yNods = len(uGridMatrix) - 1
        xNods = len(uGridMatrix[0]) - 1
        A1ResultGrid = np.zeros((yNods + 1, xNods + 1))
        for i in range(1, yNods):
            for j in range(1, xNods):
                A1ResultGrid[i][j] = (uGridMatrix[i + 1][j] - uGridMatrix[i][j]) / (self.step ** 2) - \
                                     (uGridMatrix[i][j] - uGridMatrix[i - 1][j]) / (self.step ** 2)
        return A1ResultGrid

    def A2Grid(self, uGridMatrix):
        yNods = len(uGridMatrix) - 1
        xNods = len(uGridMatrix[0]) - 1
        A2ResultGrid = np.zeros((yNods + 1, xNods + 1))
        for i in range(1, yNods):
            for j in range(1, xNods):
                A2ResultGrid[i][j] = (uGridMatrix[i][j + 1] - uGridMatrix[i][j]) / (self.step ** 2) - \
                                     (uGridMatrix[i][j] - uGridMatrix[i][j - 1]) / (self.step ** 2)
        return A2ResultGrid

    def VariableDirectionMethod(self):
        self.uGridMatrix = self.uGrid(self.xNods, self.yNods)
        self.fGridMatrix = self.fGrid(self.xNods, self.yNods)

        oldApproximation = np.zeros((self.yNods + 1, self.xNods + 1))
        for j in range(self.xNods + 1):
            oldApproximation[0][j] = self.uGridMatrix[0][j]
            oldApproximation[self.yNods][j] = self.uGridMatrix[self.yNods][j]
        for i in range(self.yNods + 1):
            oldApproximation[i][0] = self.uGridMatrix[i][0]
            oldApproximation[i][self.xNods] = self.uGridMatrix[i][self.xNods]
        self.firstApproximation = oldApproximation.copy()
        self.lastApproximation = np.zeros((self.yNods + 1, self.xNods + 1)) # Видимо не нуно хранить две последнии матрицы

        w_ = np.zeros((self.yNods + 1, self.xNods + 1))
        w = np.zeros((self.yNods + 1, self.xNods + 1))

        while not self.terminationMethod():
            #--- Вычисляем необходимые матрицы.
            F = self.LuGrid(oldApproximation) + self.fGridMatrix
            for i in range(1, self.xNods):
                for j in range(1, self.yNods):
                    w_[i][j] = (self.chi * w_[i - 1][j] + self.chi * w_[i][j - 1] + F[i][j]) / (1 + 2 * self.chi)
            for i in range(self.xNods - 1, 0, -1):
                for j in range(self.yNods - 1, 0, -1):
                    w[i][j] = (self.chi * w[i + 1][j] + self.chi * w[i][j + 1] + w_[i][j]) / (1 + 2 * self.chi)
            #---
            self.lastApproximation = self.lastApproximation + self.tau * w
            self.numberOfIterations += 1
            for j in range(self.xNods + 1):
                self.lastApproximation[0][j] = oldApproximation[0][j]
                self.lastApproximation[self.yNods][j] = oldApproximation[self.yNods][j]
            for i in range(self.yNods + 1):
                self.lastApproximation[i][0] = oldApproximation[i][0]
                self.lastApproximation[i][self.xNods] = oldApproximation[i][self.xNods]
            oldApproximation = self.lastApproximation.copy()

method = VariableDirectionMethod(0.001, 1, 1, 5, 5)
method.VariableDirectionMethod()