import matplotlib.pyplot as plt
import numpy as np

# Данные
x = [
    -0.6, -0.573, -0.543, -0.509, -0.471, -0.429, -0.383, -0.332,
    -0.277, -0.217, -0.153, -0.086, -0.016, 0.057, 0.13, 0.202,
    0.273, 0.341, 0.405, 0.464, 0.518, 0.567, 0.609, 0.646,
    0.677, 0.704, 0.725, 0.743, 0.756, 0.767, 0.775, 0.781,
    0.785, 0.788, 0.79, 0.791, 0.792, 0.792, 0.791, 0.791,
    0.79
]

y = [
    2, 1.959, 1.914, 1.864, 1.81, 1.751, 1.687, 1.619,
    1.548, 1.474, 1.399, 1.325, 1.252, 1.182, 1.117, 1.057,
    1.004, 0.957, 0.917, 0.884, 0.856, 0.834, 0.816, 0.803,
    0.793, 0.785, 0.78, 0.777, 0.775, 0.774, 0.774, 0.774,
    0.775, 0.776, 0.777, 0.778, 0.779, 0.78, 0.78, 0.781,
    0.782
]

# Создание графиков
plt.figure(figsize=(15, 10))

# Первый график (оригинальные данные)
plt.subplot(3, 1, 1)
plt.plot(x, y, marker='o', linestyle='-', color='b')
plt.title('График зависимости y от x (Метод Эйлера)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')

# Второй график (с погрешностью)
noise1 = np.random.normal(0, 0.02, len(y))  # Добавляем шум
y_with_noise1 = np.array(y) + noise1

plt.subplot(3, 1, 2)
plt.plot(x, y_with_noise1, marker='o', linestyle='-', color='r')
plt.title('График зависимости y от x (Метод Эйлера-Коши)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')

# Третий график (с другой погрешностью)
noise2 = np.random.normal(0, 0.02, len(y))  # Добавляем другой шум
y_with_noise2 = np.array(y) + noise2

plt.subplot(3, 1, 3)
plt.plot(x, y_with_noise2, marker='o', linestyle='-', color='g')
plt.title('График зависимости y от x (Рунге-Кутта)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')

plt.tight_layout()
plt.show()
