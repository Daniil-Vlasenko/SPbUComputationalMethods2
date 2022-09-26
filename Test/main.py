# import numpy as np
# import math
#
#
# def p(x, y):
#     return 1
#
#
# def q(x, y):
#     return 1
#
#
# def mu(x, y):
#     # return (x ** 3) * y + x * y * y
#     return 2 * x ** 3 * y ** 3
#
# def f(x, y):
#     # return -2 * x * (3 * y + 1)
#     return 12 * x * y * (y ** 2 + x ** 2)
#
#
# def iteration(U, N, h, F):
#     U_new = np.zeros((N + 1, N + 1))
#     for i in range(N + 1):
#         for j in range(N + 1):
#             U_new[i][j] = U[i][j]
#     for i in range(1, N):
#         for j in range(1, N):
#             U_new[i][j] = (U[i - 1][j] + U[i + 1][j] + U[i][j - 1] + U[i][j + 1] + h * h * F[i][j]) / 4
#     return U_new
#
#
# def L_h(U, N, h):
#     L = np.zeros((N + 1, N + 1))
#     for i in range(N + 1):
#         for j in range(N + 1):
#             L[i][j] = U[i][j]
#     for i in range(1, N):
#         for j in range(1, N):
#             L[i][j] = (U[i - 1][j] + U[i + 1][j] + U[i][j - 1] + U[i][j + 1] - 4 * U[i][j]) / (h * h)
#     return L
#
#
# def norm(A, N):
#     ans = -1
#     for i in range(1, N):
#         for j in range(1, N):
#             if ans < abs(A[i][j]):
#                 ans = abs(A[i][j])
#     return ans
#
#
# def do_we_continue_inf(U_i, U_0, N, h, F, eps):
#     a = np.linalg.norm(-L_h(U_i, N, h) - F, np.inf)
#     b = np.linalg.norm(-L_h(U_0, N, h) - F, np.inf)
#     c = a / b
#     if c < eps:
#         return False
#     return True
#
#
# def do_we_continue(U_i, U_0, N, h, F, eps, nev_0=None):
#     a = norm(-L_h(U_i, N, h) - F, N)
#     if nev_0:
#         b = norm(-L_h(U_0, N, h) - F, N)
#     else:
#         b = nev_0
#     c = a / b
#     if c < eps:
#         return False
#     return True
#
#
# l_x = 1
# l_y = 1
# N = 25
# M = N
# eps = 0.001
#
# h_x = l_x / N
# h_y = l_y / M
# x_vec = [i * h_x for i in range(0, N + 1)]
# y_vec = [j * h_y for j in range(0, M + 1)]
#
# Mu = np.zeros((N + 1, M + 1))
# for i in range(N + 1):
#     for j in range(M + 1):
#         Mu[i][j] = mu(x_vec[i], y_vec[j])
#
# F = np.zeros((N + 1, M + 1))
# for i in range(N + 1):
#     for j in range(M + 1):
#         F[i][j] = f(x_vec[i], y_vec[j])
#
# U_0 = np.zeros((N + 1, M + 1))
# for i in range(N + 1):
#     U_0[i][0] = mu(x_vec[i], 0)
#     U_0[i][M] = mu(x_vec[i], l_y)
# for j in range(M + 1):
#     U_0[0][j] = mu(0, y_vec[j])
#     U_0[N][j] = mu(l_x, y_vec[j])
#
# ro = math.cos(math.pi * h_x)
#
# nev_0 = norm(-L_h(U_0, N, h_x) - F, N)
#
# U = iteration(U_0, N, h_x, F)
# m = 0
# while do_we_continue(U, U_0, N, h_x, F, eps, nev_0) and m < 1000:
#     U = iteration(U, N, h_x, F)
#     m = m + 1
# t1 = norm(-L_h(U, N, h_x) - F, N)
# t2 = norm(-L_h(U_0, N, h_x) - F, N)
# print("Мера аппроксимации: ", norm(-L_h(Mu, N, h_x) - F, N))
# print("Норма невязки (k-ой): ", t1)
# print("Норма невязки нулевой: ", t2)
# print("Их частное: ", t1 / t2)
# print("Эпсилон: ", eps)
# print("Число итераций: ", m)
# print("Спектральный радиус матрицы Н: ", ro)

class IterationMethodWithOptimalParameter:
    def __init__(self):
        self.eps = 1

p = IterationMethodWithOptimalParameter()
print(p.eps)