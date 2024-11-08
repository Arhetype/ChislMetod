import numpy as np
import matplotlib.pyplot as plt

# Функция для решения системы уравнений методом Гаусса
def gauss_elimination(A, b):
    n = len(b)
    for i in range(n):
        # Прямой ход
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j] -= factor * A[i]
            b[j] -= factor * b[i]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

# Функция интерполяции полиномом Лагранжа
def lagrange(x, y, xx):
    N = len(x)
    N_res = len(xx)
    yy = np.zeros(N_res)

    for k in range(N):
        Li = np.ones(N_res)
        for j in range(N):
            if j != k:
                for i in range(N_res):
                    Li[i] = Li[i] * (xx[i] - x[j]) / (x[k] - x[j])
        yy = yy + y[k] * Li

    return yy

# Функция интерполяции кубическим сплайном
def cubic_spline(x, y, x_new):
    n = len(x) - 1
    h = np.diff(x)

    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])

    c = gauss_elimination(A.copy(), b)
    a = y[:-1]
    b_coeff = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b_coeff[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    def cubic_spline_eval(x_val):
        if x_val < x[0] or x_val > x[-1]:
            raise ValueError("x_val вне диапазона узлов.")
        for i in range(n):
            if x[i] <= x_val <= x[i + 1]:
                dx = x_val - x[i]
                return a[i] + b_coeff[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3

    return [cubic_spline_eval(xi) for xi in x_new]

# Функция для вычисления разделенных разностей
def divided_differences(x, y):
    n = len(y)
    coef = np.zeros((n, n))
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])

    return coef[0]

# Функция для вычисления значения полинома Ньютона
def newton_polynomial(x_val, x, coeffs):
    n = len(coeffs)
    result = coeffs[0]
    term = 1

    for i in range(1, n):
        term *= (x_val - x[i - 1])
        result += coeffs[i] * term

    return result

# Функция для аппроксимации многочленом второй степени
def polynomial_fit(x, y):
    n = len(x)
    A = np.vstack([np.ones(n), x, x ** 2]).T
    b = y
    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
    return coeffs

# Основной код
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
y = np.array([-0.02, 0.44, 0.51, 0.67, 0.69, 1.04, 1.1, 1.3, 1.7, 2.0, 2.1, 2.4, 2.90, 3.50, 3.99, 4.06, 4.54, 4.99, 5.36, 5.99])
xx = np.arange(1, 10.5, 0.5)

# 1. Полином Лагранжа
yy_lagrange = lagrange(x, y, xx)
print("Интерполяционный полином Лагранжа:")
for i in range(len(xx)):
    print(f"x[{xx[i]:.2f}]: {yy_lagrange[i]:.2f}")

# 2. Кубический сплайн
yy_spline = cubic_spline(x, y, xx)
print("\nИнтерполяция с помощью кубического сплайна:")
for i in range(len(xx)):
    print(f"x[{xx[i]:.2f}]: {yy_spline[i]:.2f}")

# 3. Полином Ньютона
coeffs_newton = divided_differences(x, y)
yy_newton = [newton_polynomial(xi, x, coeffs_newton) for xi in xx]
print("\nИнтерполяция с помощью полинома Ньютона:")
for i in range(len(xx)):
    print(f"x[{xx[i]:.2f}]: {yy_newton[i]:.2f}")

# 4. Аппроксимация многочленом второй степени
coeffs_poly = polynomial_fit(x, y)
print("\nАппроксимация многочленом второй степени:")
for x_val in xx:
    y_val = coeffs_poly[0] + coeffs_poly[1] * x_val + coeffs_poly[2] * x_val ** 2
    print(f"x[{x_val:.2f}]: {y_val:.2f}")

# Визуализация всех методов
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Узлы', zorder=5)
plt.plot(xx, yy_lagrange, label='Полином Лагранжа', color='blue', zorder=4)
plt.plot(xx, yy_spline, label='Кубический сплайн', color='green', zorder=3)
plt.plot(xx, yy_newton, label='Полином Ньютона', color='orange', zorder=2)
plt.plot(xx, [coeffs_poly[0] + coeffs_poly[1] * x_val + coeffs_poly[2] * x_val ** 2 for x_val in xx], 
         label='Многочлен второй степени', color='purple', zorder=1)
plt.title('Сравнение методов интерполяции и аппроксимации')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
