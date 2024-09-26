import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


c_g = 0.002
c_d = 8 * 10**(-4)
c_b = np.array([2, 3, 4, 5, 6]) * 10**(-3)


def kinetics(title):
    with open('C:\\Users\\а\\Desktop\\Химическая кинетика\\Азосочетания\\{}.txt'.format(title), 'r') as f:
        t, d = [], []
        for line in f.readlines():
            t.append(float(line.split()[0]))
            d.append(float(line.split()[1]))
    t, d = np.array(t) + 1, np.array(d)
    return [t, d]


([t0, D0], [t5, D5], [t1, D1], [t2, D2],
 [t3, D3], [t4, D4], [t_f, D_f]) = (kinetics('buffer'), kinetics(5), kinetics(1), kinetics(2),
                                    kinetics(3), kinetics(4), kinetics('d_inf'))


def graph(num):
    plt.title('$D_{}[t]$'.format(num))
    plt.ylabel("$D_{}$".format(num))
    plt.xlabel("$t$, мин")
    plt.grid(True, linestyle="--")
    plt.errorbar(kinetics(num)[0], kinetics(num)[1], label="Экспериментальные точки "
                                                           "\n (кусочно-линейная интерполяция)")
    plt.legend()
    plt.show()


plt.title('$D_i[t]$, $i = \overline{1,5}$')
plt.ylabel("$D_i$")
plt.xlabel("$t$, мин")
plt.xlim(1, 11.1)
plt.grid(True, linestyle="--")
for i in range(1, 6):
    plt.errorbar(kinetics(i)[0], kinetics(i)[1], label="{}, [NEt$_3$]$_0$ = {} мМ".format(i, i + 1))
plt.legend()
plt.show()


plt.title('$D_i[t]$ $\leq$ 2, $i = \overline{1,5}$')
plt.ylabel("$D_i$")
plt.xlabel("$t$, мин")
plt.xlim(1, 11)
plt.ylim(0, 2.1)
plt.grid(True, linestyle="--")
for i in range(1, 6):
    plt.errorbar(kinetics(i)[0], kinetics(i)[1], label="{}, [NEt$_3$]$_0$ = {} мМ".format(i, i + 1))
plt.hlines(2, 0, 11, linestyle='--', label='$D_{lim} = 2$')
plt.legend()
plt.show()


plt.title('$D_{буфер}[t]$')
plt.ylabel("$D_{буфер}$")
plt.xlabel("$t$, мин")
plt.grid(True, linestyle="--")
plt.errorbar(kinetics('buffer')[0], kinetics('buffer')[1], label="Экспериментальные точки "
                                                                 "\n (кусочно-линейная интерполяция)")
plt.legend()
plt.show()

plt.title('$D_{max}[t]$')
plt.ylabel("$D_{max}$")
plt.xlabel("$t$, мин")
plt.grid(True, linestyle="--")
plt.errorbar(kinetics('d_inf')[0], kinetics('d_inf')[1], label="Экспериментальные точки "
                                                               "\n (кусочно-линейная интерполяция)")
plt.legend()
plt.show()

D_inf = []
for value in D_f:
    if value > 4.3:
        D_inf.append(value)
D_inf, sigma_D_inf = np.mean(D_inf), np.sqrt(np.var(D_inf))
print(D_inf, sigma_D_inf)


def rel_ln_calc(num):
    [t_init, d_init] = kinetics(num)
    [t, d] = [[], []]
    for i in range(len(d_init)):
        if d_init[i] <= 2:
            t.append(t_init[i])
            d.append(d_init[i])
    t, d = np.array(t), np.array(d)
    rel_ln = np.log((D_inf - d * c_d / c_g) / (D_inf - d)) / (c_g - c_d)
    return [t, rel_ln]


def linear(x, k, b):
    return k * x + b


def k_eff_calc(num):
    t, alpha = rel_ln_calc(num)
    popt, pcov = curve_fit(linear, t, alpha, maxfev=5000)
    k_eff, beta = popt
    sigma_k_eff, sigma_beta = np.sqrt(np.diag(pcov))
    return [k_eff, sigma_k_eff, beta, sigma_beta]


print(k_eff_calc(1))
print(k_eff_calc(2))
print(k_eff_calc(3))
print(k_eff_calc(4))
print(k_eff_calc(5))


def k_eff_graph(num):
    t = rel_ln_calc(num)[0]
    plt.title('$\\frac{{1}}{{c_г - c_д}} \cdot \\alpha_{}[t]$'.format(num))
    plt.ylabel('$\\frac{{1}}{{c_г - c_д}} \cdot \\alpha_{}$, M$^{{-1}}$'.format(num))
    plt.xlabel("$t$, мин")
    plt.grid(True, linestyle="--")
    plt.errorbar(t, rel_ln_calc(num)[1], label="Расчетные значения \n (кусочно-линейная интерполяция)")
    t_sample = np.linspace(t[0], t[len(t) - 1], 10000)
    plt.plot(t_sample, linear(t_sample, k_eff_calc(num)[0], k_eff_calc(num)[2]),
             linewidth=3, label='Линейная аппроксимация $y = kx + b$ \n '
                                '$k = (%.1f \\pm %.1f)$ M$^{-1}$c$^{-1}$ \n $b = (%.1f \\pm %.1f)$ M$^{-1}$'
                                %(k_eff_calc(num)[0], k_eff_calc(num)[1], k_eff_calc(num)[2], k_eff_calc(num)[3]))
    plt.plot()
    plt.legend()
    plt.show()


k_eff_graph(1)
k_eff_graph(2)
k_eff_graph(3)
k_eff_graph(4)
k_eff_graph(5)


k_eff_values, sigma_k_eff_values = [], []
for n in range(1, 6):
    k_eff_values.append(k_eff_calc(n)[0])
    sigma_k_eff_values.append(k_eff_calc(n)[1])

k_eff_values, sigma_k_eff_values = np.array(k_eff_values), np.array(sigma_k_eff_values)

popt, pcov = curve_fit(linear, c_b, k_eff_values, sigma=sigma_k_eff_values, maxfev=5000)
k2, k3 = popt
sigma_k2, sigma_k3 = np.sqrt(np.diag(pcov))

plt.title('$k_{eff} = k_{eff}$([NEt$_3$]$_0$)')
plt.ylabel("$k_{eff}$, M$^{-1}$c$^{-1}$")
plt.xlabel("[NEt$_3$]$_0$, мM")
plt.ylim(10, 52)
plt.grid(True, linestyle="--")
plt.errorbar(c_b * 10**3, k_eff_values, fmt=".k", label="Экспериментальные точки")
plt.legend()
plt.show()
