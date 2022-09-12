import numpy as np


# Функция из 6го варианта.
def u(x, y):
    return 2 * (x * y) ** 3


def toFixed(numObj, digits=0):
    return float(f"{numObj:.{digits}f}")


l_x, l_y = 1, 1
N, M = 10, 10
h_x, h_y = l_x / N, l_y / M

w = [[(0,0) for j in range(N)] for i in range(M)]
print(w)
for i in range(N):
    for j in range(M):
        w[i][j] = (toFixed(h_x * j, 1), toFixed(h_y * i, 1))
    print(w[i])

'''
1. Как не округляя, выводить заданное число символов.
2. Реализовать алгоритм.

'''
