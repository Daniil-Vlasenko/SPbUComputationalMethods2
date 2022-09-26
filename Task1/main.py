import numpy as np


class IterationMethodWithOptimalParameter:
    def __init__(self, eps, xLength, yLength, xNods, yNods):
        self.eps = eps
        self.xLength = xLength
        self.yLength = yLength
        self.xNods = xNods
        self.yNods = yNods
        self.xStep = xLength / xNods
        self.yStep = yLength / yNods
        self.numberOfIterations = 0
        self.step = self.xStep  # Убрать это
        self.firstApproximation = None
        self.lastApproximation = None

    def uFunction(self, x, y):
        return 2 * x ** 3 * y ** 3

    def fFunction(self, x, y):
        return -12 * x * y * (y ** 2 + x ** 2)

    def terminationMethod(self, uGridMatrix):
        return np.max(np.abs(self.lastApproximation - uGridMatrix)) / np.max(np.abs(self.firstApproximation - uGridMatrix)) < self.eps

    def uGrid(self, xNods, yNods):
        return np.array([[self.uFunction(i / xNods, j / yNods) for j in range(yNods + 1)] for i in range(xNods + 1)])

    def fGrid(self, xNods, yNods):
        return np.array([[self.fFunction(i / xNods, j / yNods) for j in range(yNods + 1)] for i in range(xNods + 1)])

    def LuGrid(self, uApproximation, step):
        yNods = len(uApproximation) - 1
        xNods = len(uApproximation[0]) - 1
        LuResultGrid = np.zeros((yNods + 1, xNods + 1))
        for i in range(1, yNods):
            for j in range(1, xNods):
                LuResultGrid[i][j] = (uApproximation[i + 1][j] - uApproximation[i][j]) / (step ** 2) - \
                                     (uApproximation[i][j] - uApproximation[i - 1][j]) / (step ** 2) + \
                                     (uApproximation[i][j + 1] - uApproximation[i][j]) / (step ** 2) - \
                                     (uApproximation[i][j] - uApproximation[i][j - 1]) / (step ** 2)
        return LuResultGrid

    def iterationMethodWithOptimalParameter(self):
        uGridMatrix = self.uGrid(self.xNods, self.yNods)
        fGridMatrix = self.fGrid(self.xNods, self.yNods)

        oldApproximation = np.zeros((self.yNods + 1, self.xNods + 1))
        for j in range(self.xNods + 1):
            oldApproximation[0][j] = uGridMatrix[0][j]
            oldApproximation[self.yNods][j] = uGridMatrix[self.yNods][j]
        for i in range(self.yNods + 1):
            oldApproximation[i][0] = uGridMatrix[i][0]
            oldApproximation[i][self.xNods] = uGridMatrix[i][self.xNods]
        self.firstApproximation = oldApproximation.copy()
        self.lastApproximation = np.zeros((self.yNods + 1, self.xNods + 1))

        while not self.terminationMethod(uGridMatrix):
            self.numberOfIterations += 1
            for i in range(1, self.yNods):
                for j in range(1, self.xNods):
                    self.lastApproximation[i][j] = (oldApproximation[i - 1][j] + oldApproximation[i + 1][j] +
                                              oldApproximation[i][j - 1] + oldApproximation[i][j + 1] +
                                              self.step ** 2 * fGridMatrix[i][j]) / 4
            for j in range(self.xNods + 1):
                self.lastApproximation[0][j] = oldApproximation[0][j]
                self.lastApproximation[self.yNods][j] = oldApproximation[self.yNods][j]
            for i in range(self.yNods + 1):
                self.lastApproximation[i][0] = oldApproximation[i][0]
                self.lastApproximation[i][self.xNods] = oldApproximation[i][self.xNods]
            for i in range(self.yNods + 1):
                for j in range(self.xNods + 1):
                    oldApproximation[i][j] = self.lastApproximation[i][j]

        LuResultGrid = self.LuGrid(self.lastApproximation, self.step) 
        LuFirstGrid = self.LuGrid(self.firstApproximation, self.step)

        print("Iteration method with optimal parameter. Variant 6.")
        print("eps:", self.eps, "\n")

        print("Measure of approximation.  ||F-AU_*||:", np.max(np.abs(LuResultGrid[1:-1, 1:-1] + fGridMatrix[1:-1, 1:-1])))
        print("Discrepancy norm for U^0.  ||F-AU^0||:", np.max(np.abs(LuFirstGrid[1:-1, 1:-1] + fGridMatrix[1:-1, 1:-1])))
        print("Number of iterations:", self.numberOfIterations)

method = IterationMethodWithOptimalParameter(0.001, 1, 1, 3, 3)
method.iterationMethodWithOptimalParameter()