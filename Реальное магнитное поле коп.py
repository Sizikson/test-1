import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import mu_0, k
import warnings
warnings.filterwarnings('ignore')


class MagneticNanoparticle:
  
    
    def __init__(self, particle_type='magnetite'):
    
        
        self.T = 310.15  # 37°C
        
        
        self.eta_blood = 0.0040  
        
        
        self.rho_blood = 1060
        
        
        
        if particle_type == 'magnetite':
            
            self.Ms = 446e3 
            
            
            self.rho_particle = 5180
            
            
            self.K_anisotropy = 1.1e4
            
        elif particle_type == 'mnfe2o4':
            
            self.Ms = 330e3  
            self.rho_particle = 4900
            self.K_anisotropy = 0.8e4
            
        else:
            
            self.Ms = 400e3
            self.rho_particle = 5000
            self.K_anisotropy = 1.0e4
        
        
        self.core_diameter = 15e-9 
        
        
        self.hydro_diameter = self.core_diameter + 20e-9  
        
        # Объем магнитного ядра (м³)
        self.V_core = (4/3) * np.pi * (self.core_diameter/2)**3
        
        # Объем частицы с оболочкой (м³)
        self.V_hydro = (4/3) * np.pi * (self.hydro_diameter/2)**3
        
        # Магнитный момент частицы (А·м²)
        
        self.m = self.Ms * self.V_core
        
        # Масса частицы
        self.mass = self.rho_particle * self.V_core
        
        # Удельный магнитный момент (А·м²/кг)
        self.specific_moment = self.m / self.mass  
        
        # Эффективный коэффициент вращательного трения 
        self.eta_rot = 8 * np.pi * self.eta_blood * (self.hydro_diameter/2)**3
        
        # Время релаксации Броуна
        self.tau_B = (3 * self.eta_blood * self.V_hydro) / (k * self.T)
        
        # Время релаксации Нееля 
        tau_0 = 1e-9  
        self.tau_N = tau_0 * np.exp(self.K_anisotropy * self.V_core / (k * self.T))
        
        
        self.tau_eff = 1 / (1/self.tau_B + 1/self.tau_N)
        
        print(f"Инициализирована частица: {particle_type}")
        print(f"  Диаметр ядра: {self.core_diameter*1e9:.1f} нм")
        print(f"  Гидродинамический диаметр: {self.hydro_diameter*1e9:.1f} нм")
        print(f"  Магнитный момент: {self.m:.2e} А·м²")
        print(f"  Удельный момент: {self.specific_moment:.1f} А·м²/кг")
        print(f"  Время релаксации Броуна: {self.tau_B*1e6:.2f} мкс")
        print(f"  Время релаксации Нееля: {self.tau_N*1e9:.2f} нс")
        print(f"  Эффективное время: {self.tau_eff*1e6:.2f} мкс")
        print()



class MagneticDrugDeliverySimulation:
    
    
    def __init__(self, nanoparticle, B0=0.1, frequency=50, field_type='sinusoidal'):
      
        self.np_obj = nanoparticle
        self.B0 = B0
        self.frequency = frequency
        self.field_type = field_type
        
        # Характерное время системы = время релаксации
        self.tau = nanoparticle.tau_eff
        
        # (параметр адиабатичности)
        self.xi = frequency * self.tau
        
        print(f"Параметры поля: B0 = {B0*1000:.1f} мТл, f = {frequency} Гц")
        print(f"  Безразмерная частота ξ = f·τ = {self.xi:.3f}")
        print(f"  Режим: {'адиабатический' if self.xi < 0.1 else 'неадиабатический' if self.xi > 1 else 'переходный'}")
        print()
    
    def magnetic_field(self, t):
       
        omega = 2 * np.pi * self.frequency
        
        if self.field_type == 'sinusoidal':
            return self.B0 * np.cos(omega * t)
        elif self.field_type == 'square':
            # Прямоугольное поле (для сравнения)
            return self.B0 * np.sign(np.cos(omega * t))
        else:
            return self.B0 * np.cos(omega * t)
    
    def dtheta_dt(self, t, theta):
       
        B_t = self.magnetic_field(t)
        
        # Используем эффективное время релаксации для учета тепловых эффектов
        # В первом приближении: η_rot = m * B0 * τ_eff
        eta_effective = self.np_obj.m * self.B0 * self.np_obj.tau_eff
        
        return - (self.np_obj.m * B_t / eta_effective) * np.sin(theta)
    
    def run_simulation(self, t_max=None, theta0=np.pi/2, n_points=5000):
        
        
        if t_max is None:
            
            t_max = 10.0 / self.frequency
        
        t_eval = np.linspace(0, t_max, n_points)
        
        
        print(f"Запуск моделирования на {t_max:.4f} с ({t_max*1000:.2f} мс)...")
        
        solution = solve_ivp(
            self.dtheta_dt, 
            (0, t_max), 
            [theta0], 
            t_eval=t_eval, 
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        self.time = solution.t
        self.theta = solution.y[0]
        
        # Проекция магнитного момента на направление поля
        self.mz = np.cos(self.theta)
        
        # Энергия взаимодействия с полем 
        B_values = np.array([self.magnetic_field(t) for t in self.time])
        self.energy = -self.np_obj.m * B_values * self.mz
        
        print(f"Моделирование завершено. Точка:{len(self.time)}")
        print()
        
        return self.time, self.theta, self.mz
    
    def calculate_mean_projection(self):
        
        n_periods = 5
        period = 1.0 / self.frequency
        t_start = self.time[-1] - n_periods * period
        
        mask = self.time >= t_start
        if np.any(mask):
            return np.mean(np.abs(self.mz[mask]))
        else:
            return np.mean(np.abs(self.mz))



def analyze_frequency_response(nanoparticle, B0=0.05, frequencies=None):
   
    
    if frequencies is None:
        # Логарифмическая шкала частот от 1 Гц до 10 кГц
        frequencies = np.logspace(0, 4, 20)
    
    mean_proj = []
    xi_values = []
    
    print("=" * 60)
    print("Анализ частотной зависимости")
    print("=" * 60)
    print(f"{'Частота (Гц)':>12} {'ξ':>10} {'Режим':>12} {'<|Mz|>':>10}")
    print("-" * 60)
    
    for freq in frequencies:
        sim = MagneticDrugDeliverySimulation(nanoparticle, B0=B0, frequency=freq)
        sim.run_simulation(t_max=20.0/freq)  # 20 периодов
        mz_mean = sim.calculate_mean_projection()
        
        mean_proj.append(mz_mean)
        xi_values.append(sim.xi)
        
        # Определение режима
        if sim.xi < 0.1:
            regime = "адиабатич."
        elif sim.xi > 10:
            regime = "заморож."
        else:
            regime = "переходн."
        
        print(f"{freq:12.1f} {sim.xi:10.3f} {regime:>12} {mz_mean:10.3f}")
    
    print("=" * 60)
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.semilogx(frequencies, mean_proj, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% удержания')
    plt.axvline(x=1/nanoparticle.tau_eff, color='g', linestyle='--', alpha=0.5, 
                label=f'f_c = {1/nanoparticle.tau_eff:.1f} Гц')
    
    plt.xlabel('Частота поля (Гц)', fontsize=12)
    plt.ylabel('Средняя проекция <|cos θ|>', fontsize=12)
    plt.title('Частотная характеристика системы доставки лекарств', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return frequencies, mean_proj

def analyze_amplitude_response(nanoparticle, frequency=50, amplitudes=None):
   
    
    if amplitudes is None:
        amplitudes = np.linspace(0.001, 0.2, 20)  # от 1 до 200 мТл
    
    mean_proj = []
    
    print("=" * 60)
    print(f"Анализ амплитудной зависимости (f = {frequency} Гц)")
    print("=" * 60)
    print(f"{'B0 (мТл)':>12} {'B0 (Тл)':>12} {'<|Mz|>':>12}")
    print("-" * 60)
    
    for B0 in amplitudes:
        sim = MagneticDrugDeliverySimulation(nanoparticle, B0=B0, frequency=frequency)
        sim.run_simulation(t_max=20.0/frequency)
        mz_mean = sim.calculate_mean_projection()
        
        mean_proj.append(mz_mean)
        print(f"{B0*1000:12.1f} {B0:12.4f} {mz_mean:12.3f}")
    
    print("=" * 60)
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(amplitudes*1000, mean_proj, 'ro-', linewidth=2, markersize=8)
    plt.axhline(y=0.5, color='b', linestyle='--', alpha=0.5, label='50% удержания')
    
    plt.xlabel('Амплитуда поля (мТл)', fontsize=12)
    plt.ylabel('Средняя проекция <|cos θ|>', fontsize=12)
    plt.title('Амплитудная характеристика при f = {} Гц'.format(frequency), fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return amplitudes, mean_proj

def plot_detailed_results(sim):
    
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    
    t_ms = sim.time * 1000
    
    # 1. Угол θ(t)
    ax = axes[0, 0]
    ax.plot(t_ms, sim.theta * 180/np.pi, 'b-', linewidth=1)
    ax.set_xlabel('Время (мс)', fontsize=11)
    ax.set_ylabel('Угол θ (градусы)', fontsize=11)
    ax.set_title('Ориентация частицы', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 2. Проекция Mz = cos θ
    ax = axes[0, 1]
    ax.plot(t_ms, sim.mz, 'r-', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax.set_xlabel('Время (мс)', fontsize=11)
    ax.set_ylabel('Проекция Mz = cos θ', fontsize=11)
    ax.set_title('Удержание частицы полем', fontsize=12)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # 3. Магнитное поле B(t)
    ax = axes[0, 2]
    B_values = [sim.magnetic_field(t) for t in sim.time]
    ax.plot(t_ms, B_values, 'g-', linewidth=1)
    ax.set_xlabel('Время (мс)', fontsize=11)
    ax.set_ylabel('Поле B(t) (Тл)', fontsize=11)
    ax.set_title('Внешнее магнитное поле', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 4. Энергия взаимодействия
    ax = axes[1, 0]
    ax.plot(t_ms, sim.energy * 1e21, 'm-', linewidth=1)  # в зДж (зептоджоули)
    ax.set_xlabel('Время (мс)', fontsize=11)
    ax.set_ylabel('Энергия (зДж)', fontsize=11)
    ax.set_title('Энергия взаимодействия с полем', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 5. Гистограмма распределения углов
    ax = axes[1, 1]
    angles_deg = sim.theta * 180/np.pi
    ax.hist(angles_deg, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Угол θ (градусы)', fontsize=11)
    ax.set_ylabel('Плотность вероятности', fontsize=11)
    ax.set_title('Распределение ориентации', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 6. Фазовая диаграмма (θ vs B)
    ax = axes[1, 2]
    ax.plot(B_values, sim.theta * 180/np.pi, 'b.', markersize=1, alpha=0.5)
    ax.set_xlabel('Поле B (Тл)', fontsize=11)
    ax.set_ylabel('Угол θ (градусы)', fontsize=11)
    ax.set_title('Фазовая траектория', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Результаты моделирования: B₀ = {sim.B0*1000:.1f} мТл, f = {sim.frequency} Гц', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig



def main():
    
    
    print("=" * 70)
    print("МОДЕЛИРОВАНИЕ МАГНИТНЫХ СВОЙСТВ НОСИТЕЛЕЙ ЛЕКАРСТВ")
    print("Расчет на основе реальных физических параметров")
    print("=" * 70)
    print()
    
    # 1. Создаем наночастицу с реальными параметрами
    print("1. ИНИЦИАЛИЗАЦИЯ НАНОЧАСТИЦЫ")
    print("-" * 50)
    np_magnetite = MagneticNanoparticle(particle_type='magnetite')
    
    # 2. Базовое моделирование при типичных параметрах
    print("2. БАЗОВОЕ МОДЕЛИРОВАНИЕ")
    print("-" * 50)
    print("Типичные параметры для магнитоуправляемой доставки [citation:2][citation:4]:")
    print("  - Поле: 50-100 мТл (достижимо постоянными магнитами)")
    print("  - Частота: 50 Гц (промышленная частота) - 500 Гц")
    print()
    
    # Вариант 1: Низкая частота, среднее поле
    sim1 = MagneticDrugDeliverySimulation(np_magnetite, B0=0.05, frequency=50)
    sim1.run_simulation(t_max=0.2)  # 200 мс
    plot_detailed_results(sim1)
    
    # Вариант 2: Высокая частота (для сравнения)
    sim2 = MagneticDrugDeliverySimulation(np_magnetite, B0=0.1, frequency=500)
    sim2.run_simulation(t_max=0.04)  # 40 мс
    plot_detailed_results(sim2)
    
    # 3. Анализ частотной зависимости
    print("\n3. АНАЛИЗ ЧАСТОТНОЙ ЗАВИСИМОСТИ")
    print("-" * 50)
    freqs, proj = analyze_frequency_response(np_magnetite, B0=0.1)
    
    # 4. Анализ амплитудной зависимости
    print("\n4. АНАЛИЗ АМПЛИТУДНОЙ ЗАВИСИМОСТИ")
    print("-" * 50)
    amps, amp_proj = analyze_amplitude_response(np_magnetite, frequency=50)
    
    # 5. Сравнение разных материалов
    print("\n5. СРАВНЕНИЕ РАЗЛИЧНЫХ МАТЕРИАЛОВ")
    print("-" * 50)
    print("Сравнение магнетита и феррита марганца при 100 Гц, 50 мТл [citation:10]:")
    
    np_mnfe2o4 = MagneticNanoparticle(particle_type='mnfe2o4')
    
    sim_fe = MagneticDrugDeliverySimulation(np_magnetite, B0=0.05, frequency=100)
    sim_fe.run_simulation(t_max=0.1)
    mz_fe = sim_fe.calculate_mean_projection()
    
    sim_mn = MagneticDrugDeliverySimulation(np_mnfe2o4, B0=0.05, frequency=100)
    sim_mn.run_simulation(t_max=0.1)
    mz_mn = sim_mn.calculate_mean_projection()
    
    print(f"  Магнетит (Fe3O4):    <|Mz|> = {mz_fe:.3f}")
    print(f"  Феррит марганца:     <|Mz|> = {mz_mn:.3f}")
    
   
    print("\n" + "=" * 70)
    print("ВЫВОДЫ И РЕКОМЕНДАЦИИ ДЛЯ ЭКСПЕРИМЕНТА")
    print("=" * 70)
    
    
    for f, p in zip(freqs, proj):
        if p > 0.7:
            f_opt = f
            break
    else:
        f_opt = freqs[0]
    
    
    for a, p in zip(amps, amp_proj):
        if p > 0.7:
            B_opt = a
            break
    else:
        B_opt = amps[-1]
    
    print(f"\nОптимальные параметры для данной системы:")
    print(f"  • Частота поля: f_opt ≈ {f_opt:.0f} Гц")
    print(f"  • Амплитуда поля: B_opt ≈ {B_opt*1000:.0f} мТл")
    print(f"  • Время релаксации частицы: τ = {np_magnetite.tau_eff*1e6:.1f} мкс")
    print()
    print("Физическая интерпретация:")
    print("  • При f < 1/(2πτ) частица успевает следить за полем (адиабатический режим)")
    print("  • При f > 1/(2πτ) частица не успевает, удержание падает")
    print("  • Для эффективного управления достаточно B0 > 30-50 мТл")
    print()
    print("Практические рекомендации:")
    print("  1. Для магнитной гипертермии использовать f > 100 кГц (не наш случай)")
    print("  2. Для механического управления движением использовать f < 100 Гц")
    print("  3. Для удержания частиц в потоке крови нужно B0 > 0.1 Тл [citation:2]")
    print("  4. Покрытие частиц увеличивает гидродинамический диаметр и τ [citation:6]")
    
    plt.show()

if __name__ == "__main__":
    main()
