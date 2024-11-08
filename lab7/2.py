import numpy as np
import matplotlib.pyplot as plt
# Задаем узлы (x, y)
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
y = np.array([-0.10, -0.21, 0.01, 0.05, -0.13, -0.23, -0.21, -0.43, -0.57, -0.44, -0.44,
            -0.83, -0.78, -0.81, -1.06, -1.41, -1.40, -1.70, -1.96, -1.91])

# для вычисления коэффициентов многочлена второй степени
def polynomial_fit(x, y):
    n = len(x)

    # матрицу A и вектор b для системы уравнений
    A = np.vstack([np.ones(n), x, x ** 2]).T
    b = y

    # Решаем систему уравнений A * coeffs = b
    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
    return coeffs
# Вычисляем коэффициенты многочлена
coeffs = polynomial_fit(x, y)
print("коэффициенты аппроксимирующегомного члена второй степени")
print(coeffs)

# для вычисления значения многочлена
def polynomial(x_val, coeffs):
    return coeffs[0] + coeffs[1] * x_val + coeffs[2] * x_val ** 2

# Вывод значений многочлена
x_values = np.arange(1, 10.5, 0.5)
for x_val in x_values:
    y_val = polynomial(x_val, coeffs)
    print(f"x: {x_val:.1f}, y: {y_val:.3f}")
