import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

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

# Магнитные параметры
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
# ПОСТРОЕНИЕ И СОХРАНЕНИЕ ГРАФИКОВ (ОСНОВНОЙ РАСЧЁТ)
# =============================================================================

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

# ---- Проекции траектории на координатные плоскости ----
fig_proj, axes_proj = plt.subplots(1, 3, figsize=(15, 5))
axes_proj[0].plot(x, y, 'b-')
axes_proj[0].set_xlabel('x, м')
axes_proj[0].set_ylabel('y, м')
axes_proj[0].set_title('Проекция XY')
axes_proj[0].grid(True)
axes_proj[1].plot(x, z, 'g-')
axes_proj[1].set_xlabel('x, м')
axes_proj[1].set_ylabel('z, м')
axes_proj[1].set_title('Проекция XZ')
axes_proj[1].grid(True)
axes_proj[2].plot(y, z, 'r-')
axes_proj[2].set_xlabel('y, м')
axes_proj[2].set_ylabel('z, м')
axes_proj[2].set_title('Проекция YZ')
axes_proj[2].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'trajectory_projections.png'), dpi=150)
plt.close(fig_proj)

# ---- Графики координат, скоростей, угла и момента ----
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0,0].plot(t, x, label='x')
axes[0,0].plot(t, y, label='y')
axes[0,0].plot(t, z, label='z')
axes[0,0].set_xlabel('Время, с')
axes[0,0].set_ylabel('Координата, м')
axes[0,0].legend()
axes[0,0].grid(True)
axes[0,0].set_title('Координаты')

axes[0,1].plot(t, vx, label='vx')
axes[0,1].plot(t, vy, label='vy')
axes[0,1].plot(t, vz, label='vz')
axes[0,1].set_xlabel('Время, с')
axes[0,1].set_ylabel('Скорость, м/с')
axes[0,1].legend()
axes[0,1].grid(True)
axes[0,1].set_title('Скорости')

axes[0,2].plot(t, theta, 'm-')
axes[0,2].set_xlabel('Время, с')
axes[0,2].set_ylabel('θ, рад')
axes[0,2].grid(True)
axes[0,2].set_title('Угол ориентации магнитного момента')

axes[1,0].plot(t, mz, 'c-')
axes[1,0].set_xlabel('Время, с')
axes[1,0].set_ylabel('Mz, А·м²')
axes[1,0].grid(True)
axes[1,0].set_title('Проекция магнитного момента на ось z')

# ---- График сил ----
F_drag_z_arr = -6 * np.pi * eta * R * (vz - 0.0)
F_mag_arr = magnetic_force(mz)
F_buoy_grav_const = gravity_buoyancy_force()
axes[1,1].plot(t, F_drag_z_arr, label='F_drag_z')
axes[1,1].plot(t, F_mag_arr, label='F_mag_z')
axes[1,1].axhline(y=F_buoy_grav_const, color='k', linestyle='--', label='F_grav+boy')
axes[1,1].set_xlabel('Время, с')
axes[1,1].set_ylabel('Сила, Н')
axes[1,1].legend()
axes[1,1].grid(True)
axes[1,1].set_title('Силы, действующие по оси z')

# ---- Фазовый портрет z vs vz ----
axes[1,2].plot(z, vz, 'k-', linewidth=1)
axes[1,2].set_xlabel('z, м')
axes[1,2].set_ylabel('vz, м/с')
axes[1,2].grid(True)
axes[1,2].set_title('Фазовый портрет (z, vz)')

plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'coordinates_velocity_forces.png'), dpi=150)
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

# ---- 3D поверхность и контурный график для смещения ----
fig3 = plt.figure(figsize=(12, 5))
# 3D поверхность
ax3 = fig3.add_subplot(121, projection='3d')
surf3 = ax3.plot_surface(F_grid, A_grid, Z_max*1000, cmap='viridis', alpha=0.9,
                         linewidth=0, antialiased=True)
ax3.set_xlabel('Частота f, Гц')
ax3.set_ylabel('Амплитуда B0, Тл')
ax3.set_zlabel('Максимальное смещение по z, мм')
ax3.set_title('3D поверхность')
fig3.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, label='|z_max|, мм')
# Контурный график
ax3_cont = fig3.add_subplot(122)
contour = ax3_cont.contourf(F_grid, A_grid, Z_max*1000, levels=20, cmap='viridis')
ax3_cont.set_xlabel('Частота f, Гц')
ax3_cont.set_ylabel('Амплитуда B0, Тл')
ax3_cont.set_title('Контурная карта смещения')
fig3.colorbar(contour, ax=ax3_cont, label='|z_max|, мм')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'displacement_surface_contour.png'), dpi=150)
plt.close(fig3)

# ---- Сечения: зависимость max|z| от f при фиксированных B0 ----
fig_cut1, ax_cut1 = plt.subplots(figsize=(8,6))
B0_fixed = [0.05, 0.1, 0.2, 0.3]
for Bfix in B0_fixed:
    idx = np.argmin(np.abs(amplitudes - Bfix))
    z_cut = Z_max[idx, :] * 1000
    ax_cut1.plot(frequencies, z_cut, 'o-', label=f'B0 = {Bfix} Тл')
ax_cut1.set_xlabel('Частота f, Гц')
ax_cut1.set_ylabel('Максимальное смещение |z|, мм')
ax_cut1.legend()
ax_cut1.grid(True)
ax_cut1.set_title('Зависимость смещения от частоты при разных B0')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'displacement_vs_freq.png'), dpi=150)
plt.close(fig_cut1)

# ---- Сечения: зависимость max|z| от B0 при фиксированных f ----
fig_cut2, ax_cut2 = plt.subplots(figsize=(8,6))
f_fixed = [5, 10, 20, 40]
for ffix in f_fixed:
    idx = np.argmin(np.abs(frequencies - ffix))
    z_cut = Z_max[:, idx] * 1000
    ax_cut2.plot(amplitudes, z_cut, 's-', label=f'f = {ffix} Гц')
ax_cut2.set_xlabel('Амплитуда B0, Тл')
ax_cut2.set_ylabel('Максимальное смещение |z|, мм')
ax_cut2.legend()
ax_cut2.grid(True)
ax_cut2.set_title('Зависимость смещения от амплитуды при разных f')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'displacement_vs_B0.png'), dpi=150)
plt.close(fig_cut2)

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

# ---- 3D поверхность и контур для амплитуды Mz ----
fig4 = plt.figure(figsize=(12, 5))
ax4 = fig4.add_subplot(121, projection='3d')
surf4 = ax4.plot_surface(F_grid2, A_grid2, Mz_amp*1e7, cmap='plasma', alpha=0.9,
                         linewidth=0, antialiased=True)
ax4.set_xlabel('Частота f, Гц')
ax4.set_ylabel('Амплитуда B0, Тл')
ax4.set_zlabel('Амплитуда Mz, ×10⁻⁷ А·м²')
ax4.set_title('3D поверхность')
fig4.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10, label='ΔMz/2, ×10⁻⁷ А·м²')
ax4_cont = fig4.add_subplot(122)
contour2 = ax4_cont.contourf(F_grid2, A_grid2, Mz_amp*1e7, levels=20, cmap='plasma')
ax4_cont.set_xlabel('Частота f, Гц')
ax4_cont.set_ylabel('Амплитуда B0, Тл')
ax4_cont.set_title('Контурная карта амплитуды Mz')
fig4.colorbar(contour2, ax=ax4_cont, label='ΔMz/2, ×10⁻⁷ А·м²')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'mz_amplitude_surface_contour.png'), dpi=150)
plt.close(fig4)

# =============================================================================
# ДОПОЛНИТЕЛЬНЫЕ РАСЧЁТЫ И ПРОВЕРКИ
# =============================================================================

# 1. Число Рейнольдса для частицы
v_rel = np.sqrt((vx - U0)**2 + vy**2 + vz**2)   # относительная скорость
Re = 2 * R * rho_f * v_rel / eta
max_Re = np.max(Re)
print(f"\nМаксимальное число Рейнольдса: {max_Re:.3f}")
if max_Re < 0.5:
    print("Модель Стокса применима (Re < 0.5).")
else:
    print("Внимание: Re > 0.5, возможны отклонения от закона Стокса.")

# 2. Время оседания без магнитного поля (под действием силы тяжести и вязкости)
v_terminal = (rho_p - rho_f) * V * g / (6 * np.pi * eta * R)   # установившаяся скорость
print(f"Скорость оседания без поля: {v_terminal*1000:.2f} мм/с")
# Время падения с высоты, например, 5 мм (характерный размер сосуда)
h_fall = 5e-3   # 5 мм
if v_terminal > 0:
    t_fall = h_fall / v_terminal
    print(f"Время падения с высоты {h_fall*1000:.0f} мм без поля: {t_fall:.2f} с")

# 3. Отношение максимальной магнитной силы к силе тяжести
F_mag_max = m_mag * G
F_grav_net = abs((rho_p - rho_f) * V * g)
ratio = F_mag_max / F_grav_net
print(f"Отношение максимальной магнитной силы к эффективной силе тяжести: {ratio:.2f}")

# 4. Средняя скорость дрейфа по z за время моделирования
mean_vz = np.mean(vz)
print(f"Средняя скорость по z за {t_max} с: {mean_vz*1000:.2f} мм/с")

# 5. Коэффициент эффективности (отношение максимального смещения с полем к смещению без поля за то же время)
z_no_field = 0.5 * (F_grav_net / m_p) * t_max**2   # кинематика без учёта вязкости (грубо)
# Более точно: решим диффур без поля (только сила тяжести + вязкость)
def sys_no_mag(t, state):
    x, y, z, vx, vy, vz = state
    F_drag_z = drag_force(vz, 0.0)
    F_buoy_grav = gravity_buoyancy_force()
    az = (F_drag_z + F_buoy_grav) / m_p
    return [vx, vy, vz, 0, 0, az]
state0_no_mag = [x0, y0, z0, vx0, vy0, vz0]
sol_no_mag = solve_ivp(sys_no_mag, (0, t_max), state0_no_mag, t_eval=t_eval, method='RK45')
z_no_field_sim = sol_no_mag.y[2]
max_z_no_field = np.max(np.abs(z_no_field_sim))
eff = np.max(np.abs(z)) / max_z_no_field if max_z_no_field != 0 else np.inf
print(f"Максимальное смещение по z без поля: {max_z_no_field*1000:.2f} мм")
print(f"Коэффициент эффективности (с полем / без поля): {eff:.2f}")

# =============================================================================
# АНИМАЦИЯ ДВИЖЕНИЯ КАПСУЛЫ В ПЛОСКОСТИ XZ
# =============================================================================
print("\nСоздание анимации...")
fig_anim, ax_anim = plt.subplots(figsize=(8, 6))
ax_anim.set_xlim(np.min(x)-0.002, np.max(x)+0.002)
ax_anim.set_ylim(np.min(z)-0.002, np.max(z)+0.002)
ax_anim.set_xlabel('x, м')
ax_anim.set_ylabel('z, м')
ax_anim.set_title('Движение капсулы в плоскости XZ')
ax_anim.grid(True)

# Рисуем траекторию
line_traj, = ax_anim.plot([], [], 'b--', linewidth=0.8, alpha=0.5)
# Капсула - круг
capsule = plt.Circle((x[0], z[0]), R, color='red', alpha=0.7, label='Капсула')
# Вектор магнитного момента (стрелка)
quiver = ax_anim.quiver(x[0], z[0], 0, 0, angles='xy', scale_units='xy', scale=1, color='green', width=0.005)
ax_anim.add_patch(capsule)
ax_anim.legend([capsule, quiver], ['Капсула', 'Магнитный момент'], loc='upper right')

def init_anim():
    capsule.center = (x[0], z[0])
    line_traj.set_data([], [])
    quiver.set_offsets([x[0], z[0]])
    quiver.set_UVC(m_mag * np.cos(theta[0]), m_mag * np.sin(theta[0]))
    return capsule, line_traj, quiver

def update_anim(frame):
    capsule.center = (x[frame], z[frame])
    # обновляем траекторию
    line_traj.set_data(x[:frame+1], z[:frame+1])
    quiver.set_offsets([x[frame], z[frame]])
    # вектор момента: горизонтальная проекция m_mag * cos(theta), вертикальная m_mag * sin(theta)
    # отображаем как стрелку с длиной, пропорциональной m_mag (нормируем для наглядности)
    scale_arrow = 1e-4   # масштаб для видимости стрелки (момент ~1e-7, а размеры ~0.01 м)
    u_arrow = m_mag * np.cos(theta[frame]) / scale_arrow
    v_arrow = m_mag * np.sin(theta[frame]) / scale_arrow
    quiver.set_UVC(u_arrow, v_arrow)
    return capsule, line_traj, quiver

anim = FuncAnimation(fig_anim, update_anim, frames=len(t), init_func=init_anim,
                     interval=20, blit=True, repeat=False)

# Сохраняем анимацию в GIF (требуется pillow)
gif_path = os.path.join(current_dir, 'animation_xz.gif')
try:
    anim.save(gif_path, writer='pillow', fps=30)
    print(f"Анимация сохранена: {gif_path}")
except Exception as e:
    print(f"Не удалось сохранить GIF: {e}. Убедитесь, что установлена библиотека pillow.")
plt.close(fig_anim)

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
print("\nВсе графики и анимация сохранены в папку со скриптом:")
print(" - trajectory.png")
print(" - trajectory_projections.png")
print(" - coordinates_velocity_forces.png")
print(" - displacement_surface_contour.png")
print(" - displacement_vs_freq.png")
print(" - displacement_vs_B0.png")
print(" - mz_amplitude_surface_contour.png")
print(" - animation_xz.gif (если установлена pillow)")
