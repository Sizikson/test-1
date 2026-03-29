import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import os

# =============================================================================
# ФИЗИЧЕСКИЕ ПАРАМЕТРЫ (СИ)
# =============================================================================

# Капсула (сфера)
R = 100e-6                # радиус, м
rho_p = 2700.0            # плотность карбоната кальция, кг/м^3
V = 4/3 * np.pi * R**3    # объём, м^3
m_p = rho_p * V           # масса, кг

# Кровь
rho_f = 1060.0            # плотность крови, кг/м^3
eta = 0.0035              # динамическая вязкость, Па·с

# Магнитные параметры (используются и для полной модели, и для исследования)
m_mag = 1e-7              # магнитный момент капсулы, А·м^2
B0_main = 0.1             # амплитуда переменного поля для основного расчёта, Тл
f_main = 10.0             # частота поля для основного расчёта, Гц
G = 10.0                  # градиент магнитного поля по оси z, Тл/м
omega_main = 2 * np.pi * f_main

# Поток крови (по оси x)
U0 = 0.1                  # скорость потока, м/с

# Гравитация
g = 9.81                  # ускорение свободного падения, м/с^2

# Начальные условия
x0 = 0.0
y0 = 0.0
z0 = 0.0
vx0 = 0.0
vy0 = 0.0
vz0 = 0.0
theta0 = np.pi / 2

# Время моделирования (основное)
t_max = 5.0
n_points = 1000

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def drag_force(v, v_flow):
    """Сила вязкого трения (Стокса) для сферы"""
    return -6 * np.pi * eta * R * (v - v_flow)

def magnetic_force(mz):
    """Магнитная сила, действующая вдоль оси z (за счёт градиента)"""
    return mz * G

def gravity_buoyancy_force():
    """Суммарная сила тяжести и Архимеда (вдоль z)"""
    return (rho_p - rho_f) * V * g

# =============================================================================
# СИСТЕМА ОДУ (основная)
# =============================================================================

def system(t, state):
    x, y, z, vx, vy, vz, theta = state
    B_ac = B0_main * np.cos(omega_main * t)
    mz = m_mag * np.cos(theta)
    
    F_drag_x = drag_force(vx, U0)
    F_drag_y = drag_force(vy, 0.0)
    F_drag_z = drag_force(vz, 0.0)
    F_mag_z = magnetic_force(mz)
    F_buoy_grav = gravity_buoyancy_force()
    
    ax = F_drag_x / m_p
    ay = F_drag_y / m_p
    az = (F_drag_z + F_mag_z + F_buoy_grav) / m_p
    
    dtheta_dt = - (m_mag * B_ac * np.sin(theta)) / (8 * np.pi * eta * R**3)
    
    return [vx, vy, vz, ax, ay, az, dtheta_dt]

# =============================================================================
# ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ (основное)
# =============================================================================

t_span = (0, t_max)
t_eval = np.linspace(0, t_max, n_points)
state0 = [x0, y0, z0, vx0, vy0, vz0, theta0]

sol = solve_ivp(system, t_span, state0, t_eval=t_eval,
                method='RK45', rtol=1e-6, atol=1e-9)

t = sol.t
x = sol.y[0]
y = sol.y[1]
z = sol.y[2]
vx = sol.y[3]
vy = sol.y[4]
vz = sol.y[5]
theta = sol.y[6]
mz = m_mag * np.cos(theta)

# =============================================================================
# ПОСТРОЕНИЕ И СОХРАНЕНИЕ ГРАФИКОВ
# =============================================================================

# Получаем путь к папке, где находится скрипт
current_dir = os.path.dirname(os.path.abspath(__file__))

# ---- 3D траектория ----
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(x, y, z, 'b-', linewidth=1.5, label='Траектория')
ax1.scatter(x[0], y[0], z[0], color='green', s=100, label='Старт')
ax1.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='Финиш')
ax1.set_xlabel('x, м')
ax1.set_ylabel('y, м')
ax1.set_zlabel('z, м')
ax1.set_title('Траектория движения капсулы')
ax1.legend()
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'trajectory.png'), dpi=150)
plt.close(fig1)

# ---- Графики координат, скорости и момента ----
fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(t, x, label='x')
axes[0, 0].plot(t, y, label='y')
axes[0, 0].plot(t, z, label='z')
axes[0, 0].set_xlabel('Время, с')
axes[0, 0].set_ylabel('Координата, м')
axes[0, 0].legend()
axes[0, 0].grid(True)


plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'coordinates_velocity.png'), dpi=150)
plt.close(fig2)

# =============================================================================
# ПАРАМЕТРИЧЕСКОЕ ИССЛЕДОВАНИЕ 1: максимальное смещение по z
# =============================================================================

print("\nПараметрическое исследование 1: расчёт максимального смещения по z...")

frequencies = np.linspace(1.0, 50.0, 20)      # Гц
amplitudes = np.linspace(0.02, 0.3, 20)       # Тл
t_max_param = 5.0
n_points_param = 200

def compute_max_displacement(freq, amp):
    """Возвращает максимальное смещение по z за время t_max_param"""
    omega_loc = 2 * np.pi * freq
    B0_loc = amp

    def sys_loc(t, state):
        x, y, z, vx, vy, vz, theta = state
        B_ac = B0_loc * np.cos(omega_loc * t)
        mz = m_mag * np.cos(theta)
        F_drag_x = drag_force(vx, U0)
        F_drag_y = drag_force(vy, 0.0)
        F_drag_z = drag_force(vz, 0.0)
        F_mag_z = magnetic_force(mz)
        F_buoy_grav = gravity_buoyancy_force()
        ax = F_drag_x / m_p
        ay = F_drag_y / m_p
        az = (F_drag_z + F_mag_z + F_buoy_grav) / m_p
        dtheta_dt = - (m_mag * B_ac * np.sin(theta)) / (8 * np.pi * eta * R**3)
        return [vx, vy, vz, ax, ay, az, dtheta_dt]

    state0_loc = [float(x0), float(y0), float(z0),
                  float(vx0), float(vy0), float(vz0), float(theta0)]
    t_eval_loc = np.linspace(0, t_max_param, n_points_param)
    sol_loc = solve_ivp(sys_loc, (0, t_max_param), state0_loc,
                        t_eval=t_eval_loc, method='RK45', rtol=1e-5, atol=1e-8)
    return np.max(np.abs(sol_loc.y[2]))   # max |z|

F_grid, A_grid = np.meshgrid(frequencies, amplitudes)
Z_max = np.zeros_like(F_grid)

for i, freq in enumerate(frequencies):
    for j, amp in enumerate(amplitudes):
        Z_max[j, i] = compute_max_displacement(freq, amp)
        print(f"f={freq:.2f} Гц, B0={amp:.3f} Тл -> смещение = {Z_max[j, i]*1000:.2f} мм")

fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111, projection='3d')
surf3 = ax3.plot_surface(F_grid, A_grid, Z_max*1000, cmap='viridis', alpha=0.9,
                         linewidth=0, antialiased=True)
ax3.set_xlabel('Частота f, Гц')
ax3.set_ylabel('Амплитуда B0, Тл')
ax3.set_zlabel('Максимальное смещение по z, мм')
ax3.set_title('Зависимость смещения капсулы от параметров поля')
fig3.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, label='|z_max|, мм')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'displacement_surface.png'), dpi=150)
plt.close(fig3)

# =============================================================================
# ПАРАМЕТРИЧЕСКОЕ ИССЛЕДОВАНИЕ 2: амплитуда колебаний Mz
# =============================================================================

print("\nПараметрическое исследование 2: расчёт амплитуды Mz...")

def compute_mz_amplitude(freq, amp):
    """Решает уравнение вращения момента и возвращает амплитуду Mz = (max - min)/2"""
    omega_loc = 2 * np.pi * freq
    B0_loc = amp
    t_eval_loc = np.linspace(0, t_max_param, n_points_param)

    def ode_theta(t, theta):
        B_ac = B0_loc * np.cos(omega_loc * t)
        return - (m_mag * B_ac * np.sin(theta)) / (8 * np.pi * eta * R**3)

    sol_theta = solve_ivp(ode_theta, (0, t_max_param), [theta0],
                          t_eval=t_eval_loc, method='RK45', rtol=1e-6, atol=1e-9)
    mz_vals = m_mag * np.cos(sol_theta.y[0])
    return (np.max(mz_vals) - np.min(mz_vals)) / 2

F_grid2, A_grid2 = np.meshgrid(frequencies, amplitudes)
Mz_amp = np.zeros_like(F_grid2)

for i, freq in enumerate(frequencies):
    for j, amp in enumerate(amplitudes):
        Mz_amp[j, i] = compute_mz_amplitude(freq, amp)
        print(f"f={freq:.2f} Гц, B0={amp:.3f} Тл -> амплитуда Mz = {Mz_amp[j, i]*1e7:.2e}×10⁻⁷ А·м²")

fig4 = plt.figure(figsize=(10, 7))
ax4 = fig4.add_subplot(111, projection='3d')
surf4 = ax4.plot_surface(F_grid2, A_grid2, Mz_amp*1e7, cmap='plasma', alpha=0.9,
                         linewidth=0, antialiased=True)
ax4.set_xlabel('Частота f, Гц')
ax4.set_ylabel('Амплитуда B0, Тл')
ax4.set_zlabel('Амплитуда Mz, ×10⁻⁷ А·м²')
ax4.set_title('Амплитуда колебаний магнитного момента')
fig4.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10, label='ΔMz/2, ×10⁻⁷ А·м²')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'mz_amplitude_surface.png'), dpi=150)
plt.close(fig4)

# =============================================================================
# ВЫВОД ИНФОРМАЦИИ
# =============================================================================

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
print("="*60)
print(f"Радиус капсулы: {R*1e6:.0f} мкм")
print(f"Магнитный момент: {m_mag:.2e} А·м²")
print(f"Частота поля (основной расчёт): {f_main} Гц, амплитуда: {B0_main} Тл, градиент: {G} Тл/м")
print(f"Конечные координаты: x={x[-1]*1000:.2f} мм, y={y[-1]*1000:.2f} мм, z={z[-1]*1000:.2f} мм")
print(f"Максимальное смещение по z: {np.max(np.abs(z))*1000:.2f} мм")
print("\nВсе графики сохранены в папку со скриптом:")
print(" - trajectory.png")
print(" - coordinates_velocity.png")
print(" - displacement_surface.png")
print(" - mz_amplitude_surface.png")
