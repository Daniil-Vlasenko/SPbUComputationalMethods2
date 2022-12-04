import numpy as np
from math import log, pi, sin, cos
from tabulate import tabulate
import pandas as pd
import copy
from scipy.linalg import solve_banded, solve

eps = 1e-3
l_x = 1
l_y = 1
N = 5
M = 5
h_x = l_x / N
h_y = l_y / M

np.set_printoptions(precision=3)


def x(i):
    return (i * h_x)


def y(j):
    return (j * h_y)


def p(x, y):
    return 1


def q(x, y):
    return 1


def f(x, y):
    return -12 * x * y * (y ** 2 + x ** 2)


def mu(x, y):
    return 2 * x ** 3 * y ** 3


def norm(x):
    return np.amax(np.absolute(x))


p_min = 1
p_max = 1
q_min = 1
q_max = 1
eigenvalue_min = p_min * 4 / h_x ** 2 * \
                 sin(pi / 2 / N) ** 2 + q_min * 4 / h_y ** 2 * sin(pi / 2 / M) ** 2
eigenvalue_max = p_max * 4 / h_x ** 2 * \
                 cos(pi / 2 / N) ** 2 + q_max * 4 / h_y ** 2 * cos(pi / 2 / M) ** 2
rho_H = (eigenvalue_max - eigenvalue_min) / (eigenvalue_max + eigenvalue_min)

exact_solution_matrix = np.zeros((N + 1, M + 1))
for i in range(0, N + 1):
    for j in range(0, M + 1):
        exact_solution_matrix[i, j] = mu(x(i), y(j))

F = np.zeros((N + 1, M + 1))
for i in range(1, N):
    for j in range(1, M):
        F[i, j] = f(x(i), y(j))

eta = eigenvalue_min / eigenvalue_max
tau = 2 / (eigenvalue_max * eigenvalue_min) ** 0.5

df = pd.DataFrame(columns=['k', '||ΛU^k + F||', 'rel.d', '||U^k - U_*||',
                           'rel.error', 'U^k - U^(k-1)'])


def left(i, j):
    return tau / 2 * p(x(i - 0.5), y(j)) / h_x ** 2


def right(i, j):
    return tau / 2 * p(x(i + 0.5), y(j)) / h_x ** 2


def down(i, j):
    return tau / 2 * q(x(i), y(j - 0.5)) / h_y ** 2


def up(i, j):
    return tau / 2 * q(x(i), y(j + 0.5)) / h_y ** 2


def Lambda_1(U):
    result = np.zeros((N + 1, M + 1))
    for i in range(1, N):
        for j in range(1, M):
            l = p(x(i - 0.5), y(j)) / h_x ** 2
            r = p(x(i + 0.5), y(j)) / h_x ** 2
            result[i, j] = U[i - 1, j] * l + \
                           U[i + 1, j] * r - \
                           U[i, j] * (l + r)
    return result


def Lambda_2(U):
    result = np.zeros((N + 1, M + 1))
    for i in range(1, N):
        for j in range(1, M):
            d = q(x(i), y(j - 0.5)) / h_y ** 2
            u = q(x(i), y(j + 0.5)) / h_y ** 2
            result[i, j] = U[i, j - 1] * d + \
                           U[i, j + 1] * u - \
                           U[i, j] * (d + u)
    return result


def Lambda(U):
    return Lambda_1(U) + Lambda_2(U)


def AB_1(j):
    result = np.zeros((3, N + 1))
    result[1, 0] = 1
    result[1, N] = 1
    for i in range(1, N):
        result[0, i + 1] = -right(i, j)
        result[2, i - 1] = -left(i, j)
        result[1, i] = (1 - result[0, i + 1] - result[2, i - 1])
    return result


def AB_2(i):
    result = np.zeros((3, M + 1))
    result[1, 0] = 1
    result[1, M] = 1
    for j in range(1, M):
        result[0, j + 1] = -up(i, j)
        result[2, j - 1] = -down(i, j)
        result[1, j] = (1 - result[0, j + 1] - result[2, j - 1])
    return result


def B_1(U, j):
    result = np.zeros(N + 1)
    result[0] = U[0, j]
    result[N] = U[N, j]
    for i in range(1, N):
        result[i] = U[i, j - 1] * down(i, j) + \
                    U[i, j] * (1 - down(i, j) - up(i, j)) + \
                    U[i, j + 1] * up(i, j) + \
                    F[i, j] * tau / 2
    return result


def B_2(U, i):
    result = np.zeros(M + 1)
    result[0] = U[i, 0]
    result[M] = U[i, M]
    for j in range(1, M):
        result[j] = U[i - 1, j] * left(i, j) + \
                    U[i, j] * (1 - left(i, j) - right(i, j)) + \
                    U[i + 1, j] * right(i, j) + \
                    F[i, j] * tau / 2
    return result


def terminationMethod(U):
    return np.max(np.abs(U - exact_solution_matrix)) / np.max(
        np.abs(U_0 - exact_solution_matrix)) < eps


U = np.zeros((N + 1, M + 1))
for i in range(N + 1):
    U[i, 0] = mu(x(i), 0)
    U[i, M] = mu(x(i), l_y)
for j in range(1, M):
    U[0, j] = mu(0, y(j))
    U[N, j] = mu(l_x, y(j))

U_0 = copy.deepcopy(U)
iteration_count = 0

while True:
    U_prev = copy.deepcopy(U)
    for j in range(1, M):
        U[:, j] = solve_banded((1, 1), AB_1(j), B_1(U_prev, j))

    U_tmp = copy.deepcopy(U)
    for i in range(1, N):
        U[i, :] = solve_banded((1, 1), AB_2(i), B_2(U_tmp, i))

    iteration_count += 1

    if terminationMethod(U):
        break

print("\n1) Measure of approximation of a differential equation\n||ΛU_* + F||")
print(norm(Lambda(exact_solution_matrix) + F))

print("\n2) Norm of the zero approximation discrepancy\n||ΛU^0 + F|| = ||F||")
print(norm(Lambda(U_0) + F))

print("Number of iterations")
print(iteration_count)

print("\n5) Approximate solution")
print(tabulate(U.transpose()))

print("\n6) Exact solution")
print(tabulate(exact_solution_matrix.transpose()))

# print(tabulate((exact_solution_matrix - U).transpose()))
