import numpy as np


# Функция из 6го варианта.
def uFunction(x, y):
    return 2 * x ** 3 * y ** 3


def fFunction(x, y):
    return -12 * x * y * (y ** 2 + x ** 2)


def terminationMethod(fistApproximation, lastApproximation, uGrid, eps):
    return np.max(np.abs(lastApproximation - uGrid)) / np.max(np.abs(fistApproximation - uGrid)) < eps


def uGrid(xNods, yNods):
    return np.array([[uFunction(i / xNods, j / yNods) for j in range(yNods + 1)] for i in range(xNods + 1)])


def fGrid(xNods, yNods):
    return np.array([[fFunction(i / xNods, j / yNods) for j in range(yNods + 1)] for i in range(xNods + 1)])


def LuGrid(uApproximation, step):
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


# -------------------------------------------------------------------------------------------------------------------- #
eps = 0.001
xlength, yLength = 1, 1
xNods, yNods = 3, 3
xStep, yStep = xlength / xNods, yLength / yNods
# Предположение в вычислениях: h = h_x = h_y.
step = xStep
numberOfIterations = 0

uGrid = uGrid(xNods, yNods)
fFunctionGrid = fGrid(xNods, yNods)

oldApproximation = np.zeros((yNods + 1, xNods + 1))
for j in range(xNods + 1):
    oldApproximation[0][j] = uGrid[0][j]
    oldApproximation[yNods][j] = uGrid[yNods][j]
for i in range(yNods + 1):
    oldApproximation[i][0] = uGrid[i][0]
    oldApproximation[i][xNods] = uGrid[i][xNods]
fistApproximation = oldApproximation.copy()
newApproximation = np.zeros((yNods + 1, xNods + 1))

while not terminationMethod(fistApproximation, newApproximation, uGrid, eps):
    numberOfIterations += 1
    for i in range(1, yNods):
        for j in range(1, xNods):
            newApproximation[i][j] = (oldApproximation[i - 1][j] + oldApproximation[i + 1][j] +
                                      oldApproximation[i][j - 1] + oldApproximation[i][j + 1] +
                                      step ** 2 * fFunctionGrid[i][j]) / 4
    for j in range(xNods + 1):
        newApproximation[0][j] = oldApproximation[0][j]
        newApproximation[yNods][j] = oldApproximation[yNods][j]
    for i in range(yNods + 1):
        newApproximation[i][0] = oldApproximation[i][0]
        newApproximation[i][xNods] = oldApproximation[i][xNods]
    for i in range(yNods + 1):
        for j in range(xNods + 1):
            oldApproximation[i][j] = newApproximation[i][j]

LuResultGrid = LuGrid(newApproximation, step)
LuFirstGrid = LuGrid(fistApproximation, step)

print("Iteration method with optimal parameter. Variant 6.")
print("eps:", eps, "\n")

print("Measure of approximation.  ||F-AU_*||:", np.max(np.abs(LuResultGrid[1:-1, 1:-1] + fFunctionGrid[1:-1, 1:-1])))
print("Discrepancy norm for U^0.  ||F-AU^0||:", np.max(np.abs(LuFirstGrid[1:-1, 1:-1] + fFunctionGrid[1:-1, 1:-1])))
print("Number of iterations:", numberOfIterations)
