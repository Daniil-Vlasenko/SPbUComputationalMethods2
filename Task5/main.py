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
        self.length = self.xLength  # !
        self.xNods = xNods
        self.yNods = yNods
        self.nods = self.xNods  # !
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
        return self.numberOfIterations > 100
        # return np.max(np.abs(self.lastApproximation - self.uGridMatrix)) / np.max(
        #     np.abs(self.firstApproximation - self.uGridMatrix)) < self.eps

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

    def solvingSystemsTridiagonalMatrix(self, A, B, C, G):
        s = [C[0] / B[0]]
        t = [-G[0] / B[0]]
        for i in range(1, len(B) - 1):
            s.append(C[i] / (B[i] - A[i - 1] * s[i - 1]))
            t.append((A[i - 1] * t[i - 1] - G[i]) / (B[i] - A[i - 1] * s[i - 1]))
        y = [t[-1]]
        for i in range(len(B) - 2, -1, -1):
            y.append(s[i] * y[len(B) - 2 - i] + t[i])
        y.reverse()
        return y


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
        self.lastApproximation = np.zeros(
            (self.yNods + 1, self.xNods + 1))  # Видимо не нуно хранить две последнии матрицы

        intermApproximation = np.zeros((self.yNods + 1, self.xNods + 1))

        while not self.terminationMethod():
            # --- Решаем системы уравнений
            G1 = -oldApproximation - self.tau / 2 * (self.A2Grid(oldApproximation) + self.fGridMatrix)
            diag_1 = [self.A for i in range(self.nods - 1)]
            diag_1.append(0)
            diag_2 = [-self.B for i in range(self.nods - 1)]
            diag_2.insert(0, 1)
            diag_2.append(1)
            diag_3 = [self.C for i in range(self.nods - 1)]
            diag_3.insert(0, 0)
            for j in range(1, self.nods):
                G1Vector = G1[1:self.nods, j].tolist()
                G1Vector.insert(0, oldApproximation[0][j])
                G1Vector.append(oldApproximation[self.nods][j])
                u_j = self.solvingSystemsTridiagonalMatrix(diag_1, diag_2, diag_3, G1Vector)
                intermApproximation[:, j] = u_j.copy()
            intermApproximation[:, 0] = oldApproximation[:, 0].copy()
            intermApproximation[:, self.nods] = oldApproximation[:, self.nods].copy()
            # ---
            G2 = -intermApproximation - self.tau / 2 * (self.A1Grid(intermApproximation) + self.fGridMatrix)
            for i in range(1, self.nods):
                G2Vector = G2[i, 1:self.nods].tolist()
                G2Vector.insert(0, oldApproximation[i][0])
                G2Vector.append(oldApproximation[i][self.nods])
                u_i = self.solvingSystemsTridiagonalMatrix(diag_1, diag_2, diag_3, G2Vector)
                self.lastApproximation[i, :] = u_i.copy()
            self.lastApproximation[0, :] = oldApproximation[0, :].copy()
            self.lastApproximation[self.nods, :] = oldApproximation[self.nods, :].copy()
            # ---
            self.numberOfIterations += 1
            oldApproximation = self.lastApproximation.copy()


method = VariableDirectionMethod(0.001, 1, 1, 5, 5)
method.VariableDirectionMethod()
