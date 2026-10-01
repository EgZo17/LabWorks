import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. ПАРАМЕТРЫ МОДЕЛИ
D = 4.1  # Диаметр втулки, см
R = D / 2.0  # Радиус, см
H = 10.0  # Высота втулки, см

# Критический угол (в радианах)
theta_c = np.arctan(2 * R / H)

# Теоретические вероятности
p1_theory = (1 - np.cos(theta_c)) / 2
p2_theory = (1 - np.cos(theta_c)) / 2
p3_theory = np.cos(theta_c)

print("=" * 60)
print("ТЕОРЕТИЧЕСКИЕ ВЕРОЯТНОСТИ")
print("=" * 60)
print(f"Критический угол θc = {np.degrees(theta_c):.2f}°")
print(f"p1 (нижний торец)  = {p1_theory:.4f}  ({p1_theory * 100:.2f}%)")
print(f"p2 (верхний торец) = {p2_theory:.4f}  ({p2_theory * 100:.2f}%)")
print(f"p3 (бок)           = {p3_theory:.4f}  ({p3_theory * 100:.2f}%)")
print(f"Сумма              = {p1_theory + p2_theory + p3_theory:.4f}")
print()


# 2. ФУНКЦИЯ ОДНОГО ИСПЫТАНИЯ
def simulate_trial():
    """
    Генерирует случайную ориентацию втулки в 3D (равномерно на сфере)
    и возвращает номер исхода: 1 (нижний торец), 2 (верхний торец) или 3 (бок).
    """
    # Равномерное распределение направления на единичной сфере (метод Мюллера)
    u = np.random.normal(size=3)
    u = u / np.linalg.norm(u)

    # Угол между осью втулки и вертикалью (ось Z)
    theta = np.arccos(np.clip(u[2], -1.0, 1.0))

    # Геометрическое правило первого касания
    if theta < theta_c:
        return 1  # Ось близка к +Z -> нижний торец внизу
    elif theta > np.pi - theta_c:
        return 2  # Ось близка к -Z -> верхний торец внизу
    else:
        return 3  # Большой наклон -> касание боковой поверхностью


# 3. СЕРИЯ КОМПЬЮТЕРНЫХ ЭКСПЕРИМЕНТОВ
N_max = 100000
np.random.seed(42)  # Для воспроизводимости результатов
results = np.array([simulate_trial() for _ in range(N_max)])

n_array = np.arange(1, N_max + 1)
cumsum_1 = np.cumsum(results == 1) / n_array
cumsum_2 = np.cumsum(results == 2) / n_array
cumsum_3 = np.cumsum(results == 3) / n_array

# Вывод частот для характерных значений N
print("=" * 60)
print("НАКОПЛЕННЫЕ ОТНОСИТЕЛЬНЫЕ ЧАСТОТЫ КОМПЬЮТЕРНОЙ МОДЕЛИ")
print("=" * 60)
print(f"{'N':>10} | {'ω1':>8} | {'ω2':>8} | {'ω3':>8}")
print("-" * 45)
for N in [10, 100, 1000, 10000, 100000]:
    idx = N - 1
    print(f"{N:>10} | {cumsum_1[idx]:>8.4f} | {cumsum_2[idx]:>8.4f} | {cumsum_3[idx]:>8.4f}")
print()

# 4. ГРАФИК СХОДИМОСТИ
plt.figure(figsize=(11, 7))

plt.plot(n_array, cumsum_1, label='Частота ω1 (нижний торец)', alpha=0.7, linewidth=1, color='blue')
plt.plot(n_array, cumsum_2, label='Частота ω2 (верхний торец)', alpha=0.7, linewidth=1, color='orange')
plt.plot(n_array, cumsum_3, label='Частота ω3 (бок)', alpha=0.7, linewidth=1, color='green')

# Теоретические вероятности (пунктир)
plt.axhline(p1_theory, color='blue', linestyle='--', linewidth=2, label=f'Теор. p1 = {p1_theory:.3f}')
plt.axhline(p2_theory, color='orange', linestyle='--', linewidth=2, label=f'Теор. p2 = {p2_theory:.3f}')
plt.axhline(p3_theory, color='green', linestyle='--', linewidth=2, label=f'Теор. p3 = {p3_theory:.3f}')

# Физические частоты (точечный пунктир)
plt.axhline(0.08, color='blue', linestyle=':', linewidth=2.5, alpha=0.6, label='Физ. ω1 = 0.080 (N=100)')
plt.axhline(0.09, color='orange', linestyle=':', linewidth=2.5, alpha=0.6, label='Физ. ω2 = 0.090 (N=100)')
plt.axhline(0.83, color='green', linestyle=':', linewidth=2.5, alpha=0.6, label='Физ. ω3 = 0.830 (N=100)')

plt.xscale('log')
plt.xlim(10, 100000)
plt.ylim(0, 1.05)
plt.xlabel('Количество компьютерных испытаний (n)')
plt.ylabel('Накопленная относительная частота')
plt.title(
    'Сходимость относительных частот компьютерной модели\nк теоретическим вероятностям и физическому эксперименту')
plt.legend(loc='center right', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight')
print("График сходимости сохранён: convergence_plot.png")
plt.show()


# 5. ВИЗУАЛИЗАЦИЯ МОМЕНТА ПЕРВОГО КАСАНИЯ
def draw_cylinder(outcome_type, ax, title):
    """Рисует втулку, касающуюся поверхности в точке (0,0)."""
    ax.clear()

    # Горизонтальная поверхность
    ax.axhline(0, color='black', linewidth=2.5, label='Горизонтальная поверхность')

    # Выбираем угол наклона оси к вертикали
    if outcome_type == 'side':
        theta_display = theta_c + 0.4  # Гарантированно бок
        color = 'green'
        label_outcome = 'ω3 (бок)'
    else:
        theta_display = theta_c * 0.5  # Гарантированно торец
        color = 'blue'
        label_outcome = 'ω1/ω2 (торец)'

    # Угол от оси X (для матрицы поворота)
    angle_from_x = np.pi / 2 - theta_display
    cos_a = np.cos(angle_from_x)
    sin_a = np.sin(angle_from_x)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Четыре угла прямоугольника в локальных координатах (центр в (0,0))
    corners_local = np.array([
        [H / 2, R],
        [H / 2, -R],
        [-H / 2, -R],
        [-H / 2, R]
    ])

    # Поворот
    corners_rotated = corners_local @ rotation_matrix.T

    # Сдвиг вверх, чтобы самая нижняя точка оказалась на z=0
    min_z = np.min(corners_rotated[:, 1])
    corners_final = corners_rotated.copy()
    corners_final[:, 1] -= min_z

    # Рисуем втулку
    polygon = patches.Polygon(corners_final, closed=True,
                              linewidth=2, edgecolor=color,
                              facecolor=color, alpha=0.3,
                              label=f'Втулка ({label_outcome})')
    ax.add_patch(polygon)

    # Ось цилиндра (от центра нижнего торца к центру верхнего)
    axis_local = np.array([[-H / 2, 0], [H / 2, 0]])
    axis_rotated = axis_local @ rotation_matrix.T
    axis_final = axis_rotated.copy()
    axis_final[:, 1] -= min_z
    ax.plot(axis_final[:, 0], axis_final[:, 1], 'm-', linewidth=2, label='Ось цилиндра')

    # Критический угол (красный пунктир из точки касания)
    ax.plot([0, 6 * np.sin(theta_c)], [0, 6 * np.cos(theta_c)],
            'r--', linewidth=2, label=f'Критический угол θc ≈ {np.degrees(theta_c):.1f}°')

    # Вертикаль
    ax.plot([0, 0], [0, 6], 'k:', alpha=0.5, linewidth=2, label='Вертикаль (ось Z)')

    # Точка первого касания
    ax.plot(0, 0, 'ko', markersize=10, zorder=5, label='Точка первого касания')

    # Настройка осей
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
draw_cylinder('side', ax1, 'Момент первого касания: исход ω3 (бок)\nУгол наклона оси > θc')
draw_cylinder('end', ax2, 'Момент первого касания: исход ω1/ω2 (торец)\nУгол наклона оси < θc')

plt.tight_layout()
plt.savefig('visualization_first_touch.png', dpi=300, bbox_inches='tight')
print("Визуализация сохранена: visualization_first_touch.png")
plt.show()

# 6. СРАВНИТЕЛЬНАЯ ТАБЛИЦА
print("=" * 75)
print("СРАВНЕНИЕ: ТЕОРИЯ / ФИЗИКА (N=100) / КОМПЬЮТЕР (N=100000)")
print("=" * 75)
print(f"{'Исход':<25} | {'Теория':>8} | {'Физика':>8} | {'Компьютер':>10} | {'Δ физ-теор':>10}")
print("-" * 75)
p_phys = [0.08, 0.09, 0.83]
p_sim = [cumsum_1[-1], cumsum_2[-1], cumsum_3[-1]]
p_theo = [p1_theory, p2_theory, p3_theory]
labels = ['ω1 (нижний торец)', 'ω2 (верхний торец)', 'ω3 (бок)']
for i in range(3):
    delta = abs(p_phys[i] - p_theo[i])
    print(f"{labels[i]:<25} | {p_theo[i]:>8.4f} | {p_phys[i]:>8.3f} | {p_sim[i]:>10.4f} | {delta:>10.4f}")
print("=" * 75)
