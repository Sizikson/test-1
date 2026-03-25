import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# --- Параметры системы ---
m = 1.0          # Магнитный момент (нормированный)
eta = 1.0        # Вязкость (нормированная)
t_max = 10.0     # Максимальное время наблюдения
n_points = 500   # Количество точек по времени
theta0 = np.pi / 2  # Начальный угол (90 градусов)

# --- Функция для решения уравнения движения ---
def solve_particle(frequency, amplitude):
    """
    Решает уравнение движения для заданных частоты и амплитуды.
    Возвращает временную сетку и проекцию mz = cos(theta)
    """
    t_eval = np.linspace(0, t_max, n_points)
    
    def ode_func(t, theta):
        B_t = amplitude * np.cos(2 * np.pi * frequency * t)
        return - (m * B_t / eta) * np.sin(theta)
    
    solution = solve_ivp(ode_func, (0, t_max), [theta0], t_eval=t_eval, method='RK45')
    return solution.t, np.cos(solution.y[0])

# --- 3D ГРАФИК 1: Зависимость Mz от времени и частоты ---
print("Построение 3D графика 1: Mz(t, f)...")

# Создаем сетку параметров
frequencies = np.linspace(0.1, 5.0, 30)  # 30 значений частоты
B0_fixed = 2.0  # Фиксированная амплитуда

# Подготавливаем данные для 3D поверхности
T, F = np.meshgrid(np.linspace(0, t_max, n_points), frequencies)
Z = np.zeros_like(T)

for i, freq in enumerate(frequencies):
    _, mz = solve_particle(freq, B0_fixed)
    Z[i, :] = mz

# Создаем фигуру с тремя 3D подграфиками
fig = plt.figure(figsize=(18, 6))

# График 1: Mz(t, f)
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(T, F, Z, cmap='RdYlBu_r', alpha=0.9, linewidth=0, antialiased=True)
ax1.set_xlabel('Время (норм.)')
ax1.set_ylabel('Частота f (Гц)')
ax1.set_zlabel('Проекция Mz')
ax1.set_title(f'Зависимость Mz от времени и частоты\n(B0 = {B0_fixed})')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='Mz')
ax1.view_init(elev=25, azim=-45)

# --- 3D ГРАФИК 2: Зависимость Mz от времени и амплитуды ---
print("Построение 3D графика 2: Mz(t, B0)...")

amplitudes = np.linspace(0.2, 5.0, 30)  # 30 значений амплитуды
f_fixed = 1.0  # Фиксированная частота

# Подготавливаем данные
T2, A = np.meshgrid(np.linspace(0, t_max, n_points), amplitudes)
Z2 = np.zeros_like(T2)

for i, amp in enumerate(amplitudes):
    _, mz = solve_particle(f_fixed, amp)
    Z2[i, :] = mz

ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(T2, A, Z2, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
ax2.set_xlabel('Время (норм.)')
ax2.set_ylabel('Амплитуда B0')
ax2.set_zlabel('Проекция Mz')
ax2.set_title(f'Зависимость Mz от времени и амплитуды\n(f = {f_fixed} Гц)')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Mz')
ax2.view_init(elev=25, azim=-45)

# --- 3D ГРАФИК 3: Амплитуда колебаний Mz в зависимости от f и B0 ---
print("Построение 3D графика 3: Амплитуда колебаний Mz(f, B0)...")

# Создаем сетку параметров
f_grid, B0_grid = np.meshgrid(frequencies, amplitudes)
amp_oscillation = np.zeros_like(f_grid)

# Для каждой пары (f, B0) вычисляем размах колебаний Mz
for i, freq in enumerate(frequencies):
    for j, amp in enumerate(amplitudes):
        _, mz = solve_particle(freq, amp)
        # Амплитуда колебаний = (max - min) / 2
        amp_oscillation[j, i] = (np.max(mz) - np.min(mz)) / 2

ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(f_grid, B0_grid, amp_oscillation, cmap='plasma', alpha=0.9, linewidth=0, antialiased=True)
ax3.set_xlabel('Частота f (Гц)')
ax3.set_ylabel('Амплитуда B0')
ax3.set_zlabel('Амплитуда колебаний Mz')
ax3.set_title('Амплитуда отклика частицы\nв зависимости от параметров поля')
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, label='Размах Mz')
ax3.view_init(elev=30, azim=-135)

plt.tight_layout()
plt.show()

# --- Дополнительный график: Траектория движения в 3D пространстве ---
print("\nПостроение траектории движения магнитного момента в 3D...")

# Создаем отдельную фигуру для траектории
fig2 = plt.figure(figsize=(10, 8))
ax_traj = fig2.add_subplot(111, projection='3d')

# Рассчитываем для конкретных параметров (средние значения)
f_traj = 0.8
B0_traj = 2.5
t_traj, mz_traj = solve_particle(f_traj, B0_traj)

# Преобразуем mz в компоненты вектора магнитного момента
# Предполагаем, что вектор вращается в плоскости XZ
theta_actual = np.arccos(np.clip(mz_traj, -1, 1))  # Текущий угол
mx_traj = np.sin(theta_actual) * np.cos(2 * np.pi * f_traj * t_traj)  # Компонента X
my_traj = np.sin(theta_actual) * np.sin(2 * np.pi * f_traj * t_traj)  # Компонента Y (для 3D эффекта)
mz_traj_component = np.cos(theta_actual)  # Компонента Z

# Строим 3D траекторию конца вектора магнитного момента
ax_traj.plot(mx_traj, my_traj, mz_traj_component, 'b-', linewidth=1, alpha=0.7, label='Траектория момента')
ax_traj.scatter(mx_traj[0], my_traj[0], mz_traj_component[0], color='green', s=100, label='Старт')
ax_traj.scatter(mx_traj[-1], my_traj[-1], mz_traj_component[-1], color='red', s=100, label='Финиш')

# Добавляем сферу для наглядности (единичная сфера)
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax_traj.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2, linewidth=0.5)

# Настройка графика
ax_traj.set_xlabel('Mx')
ax_traj.set_ylabel('My')
ax_traj.set_zlabel('Mz')
ax_traj.set_title(f'3D траектория магнитного момента\n(f={f_traj} Гц, B0={B0_traj})')
ax_traj.legend()
ax_traj.set_xlim([-1.2, 1.2])
ax_traj.set_ylim([-1.2, 1.2])
ax_traj.set_zlim([-1.2, 1.2])
ax_traj.grid(True)

plt.tight_layout()
plt.show()

# --- Анализ результатов ---
print("\n" + "="*60)
print("АНАЛИЗ 3D ВИЗУАЛИЗАЦИИ")
print("="*60)

# Находим критические параметры
critical_point = np.unravel_index(np.argmax(amp_oscillation > 0.5), amp_oscillation.shape)
if critical_point:
    print(f"\nПереходный режим наблюдается при:")
    print(f"  Частота ~ {frequencies[critical_point[1]]:.2f} Гц")
    print(f"  Амплитуда ~ {amplitudes[critical_point[0]]:.2f}")
    print("\nФизическая интерпретация:")
    print("- Область высоких значений Mz (красный цвет на 1-м графике): частица успевает следить за полем")
    print("- Область низких значений Mz (синий цвет): частица не успевает реагировать")
    print("- Переходная область (белый/зеленый): критический режим управления")

print("\nРекомендации для экспериментов:")
print("1. Для точного позиционирования носителя: f < 1.0 Гц, B0 > 2.0")
print("2. Для нагрева (гипертермии): f > 3.0 Гц, B0 средние значения")
print("3. Для удержания в потоке: подбирать параметры из переходной области")
