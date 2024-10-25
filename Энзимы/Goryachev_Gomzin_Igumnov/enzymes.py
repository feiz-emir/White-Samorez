import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt

eps, l = 3 * 10**3, 1


def kinetics(title):
    with open('C:\\Users\\а\\Desktop\\Химическая кинетика\\Энзимы\\Первичные данные\\{}.txt'.format(title), 'r') as f:
        t, d = [], []
        for line in f.readlines():
            t.append(float(line.split()[0]))
            d.append(float(line.split()[1]))
    t, d = np.array(t), np.array(d)
    return [t, d]


[t0, D0], [t1, D1], [t2, D2], \
[t3, D3], [t4, D4], [t5, D5], \
[t6, D6], [t7, D7], [t8, D8] = kinetics(0), kinetics('глюкоза_1'), kinetics('глюкоза_2'), \
                               kinetics('ксилоза_1'), kinetics('ксилоза_2'), kinetics('глюкоза_3'), \
                               kinetics('глюкоза_4'), kinetics('манноза'), kinetics('ксилоза_3')

data = [[t0, D0], [t1, D1], [t2, D2],
        [t3, D3], [t4, D4], [t5, D5],
        [t6, D6], [t7, D7], [t8, D8]]


def init_select(num, border_value):
    t, d = data[num]
    t_init, d_init = [], []
    for i in range(len(d)):
        if d[i] <= border_value:
            t_init.append(t[i])
            d_init.append(d[i])
    t_init, d_init = np.array(t_init), np.array(d_init)
    return [t_init, d_init]


t1_init, D1_init = init_select(1, 0.3)
t2_init, D2_init = init_select(2, 0.3)
t6_init, D6_init = init_select(6, 0.5)
t7_init, D7_init = init_select(7, 0.5)


def linear(p, k0, b0):
    return k0 * p + b0


def normal_approx(p, p0, p_d, y0, y_inf):
    y = y_inf * norm.cdf(p, p0, p_d) + y0
    return y


def normal_density(p, p0, p_d, y_inf):
    y = y_inf * norm.pdf(p, p0, p_d)
    return y


def init_curve_fit(num, border_value):
    t_init, d_init = init_select(num, border_value)

    popt, pcov = curve_fit(normal_approx, t_init, d_init, maxfev=500000)
    tau, T, d0, d_inf = popt
    sigma_tau, sigma_T, sigma_d0, sigma_d_inf = np.sqrt(np.diag(pcov))

    # print('Нормальное распределение (начальный участок кривой {}):'.format(num), '\n',
    # 'tau = ', tau, '\pm', sigma_tau, '\n',
    # 'T = ', T, '\pm', sigma_T, '\n',
    # 'd0 = ', d0, '\pm', sigma_d0, '\n',
    # 'd_inf = ', d_inf, '\pm', sigma_d_inf, '\n')

    return [[tau, sigma_tau], [T, sigma_T], [d0, sigma_d0], [d_inf, sigma_d_inf]]


def init_curve_graph(num, border_value, title):
    t, d = data[num]
    t_init, d_init = init_select(num, border_value)

    [[tau, sigma_tau], [T, sigma_T], [d0, sigma_d0], [d_inf, sigma_d_inf]] = \
        init_curve_fit(num, border_value)

    plt.title('$D[t]$ ({})'.format(title), fontsize=14)
    plt.ylabel("$D$", fontsize=12)
    plt.xlabel("$t$, мин", fontsize=12)
    plt.grid(True, linestyle="--")
    plt.errorbar(t[:len(t_init) + 3], d[:len(d_init) + 3], label='Экспериментальные точки \n '
                                                                 '(кусочно-линейная интерполяция)')
    t_sample = np.linspace(t_init[0], t_init[len(t_init) - 1], 1000)
    plt.plot(t_sample, normal_approx(t_sample, tau, T, d0, d_inf),
             label='Аппроксимация по эмпирической формуле: \n '
                   '$D(t) = D_0 + D_{\infty} \cdot \int_{-\infty}^{t} \\frac{1}{\sqrt{2\pi T^2}} \cdot '
                   'exp \\left[-\\frac{(\\tau - \\tau_0)^2}{2T^2} \\right] d\\tau$ \n '
                   '$D_{\infty} = %.4f \\pm %.4f$; $D_0 = %.4f \\pm %.4f$ \n '
                   '$\\tau_0 = (%.3f \\pm %.3f)$ мин; $T = (%.3f \\pm %.3f)$ мин \n'
                   % (d_inf, sigma_d_inf, d0, sigma_d0, tau, sigma_tau, T, sigma_T))
    plt.vlines(tau, 0, 0.5 * d_inf + d0, linestyle='dashed')
    plt.legend()
    plt.show()


def linear_approx(num):
    t, d = data[num]

    popt, pcov = curve_fit(linear, t, d, maxfev=50000)
    k0, b0 = popt
    sigma_k0, sigma_b0 = np.sqrt(np.diag(pcov))

    return [[k0, sigma_k0], [b0, sigma_b0]]


[k3, sigma_k3], [b3, sigma_b3] = linear_approx(3)
[k4, sigma_k4], [b4, sigma_b4] = linear_approx(4)
[k8, sigma_k8], [b8, sigma_b8] = linear_approx(8)


plt.title('$D_{буфер}[t]$', fontsize=14)
plt.ylabel("$D_{буфер}$, 10$^{-3}$", fontsize=12)
plt.xlabel("$t$, мин", fontsize=12)
plt.grid(True, linestyle="--")
plt.errorbar(t0, D0 * 10 ** 3, label="Экспериментальные точки "
                                     "\n (кусочно-линейная интерполяция)")
plt.legend()
plt.show()


init_curve_graph(6, 0.5, 'глюкоза, 4')
init_curve_graph(7, 0.5, 'манноза')
init_curve_graph(1, 0.4, 'глюкоза, 1')
init_curve_graph(2, 0.4, 'глюкоза, 2')
init_curve_graph(5, 0.4, 'глюкоза, 3')


plt.title('$D[t]$', fontsize=14)
plt.ylabel("$D$", fontsize=12)
plt.xlabel("$t$, мин", fontsize=12)
plt.grid(True, linestyle="--")
plt.errorbar(t1, D1, label="Глюкоза, 1")
plt.errorbar(t2, D2, label="Глюкоза, 2")
plt.errorbar(t3, D3, label="Ксилоза, 1")
plt.errorbar(t4, D4, label="Ксилоза, 2")
plt.errorbar(t5, D5, label="Глюкоза, 3")
plt.errorbar(t6, D6, label="Глюкоза, 4")
plt.errorbar(t7, D7, label="Манноза")
plt.errorbar(t8, D8, label="Ксилоза, 3")
plt.legend()
plt.show()

plt.title('$D[t]$ (старый энзим)', fontsize=14)
plt.ylabel("$D$", fontsize=12)
plt.xlabel("$t$, мин", fontsize=12)
plt.grid(True, linestyle="--")
plt.errorbar(t1, D1, label="Глюкоза, 1")
plt.errorbar(t2, D2, label="Глюкоза, 2")
plt.errorbar(t3, D3, label="Ксилоза, 1")
plt.errorbar(t4, D4, label="Ксилоза, 2")
plt.errorbar(t5, D5, label="Глюкоза, 3")
plt.legend()
plt.show()


plt.title('$D[t]$ (свежий энзим)', fontsize=14)
plt.ylabel("$D$", fontsize=12)
plt.xlabel("$t$, мин", fontsize=12)
plt.grid(True, linestyle="--")
plt.errorbar(t6, D6, label="Глюкоза, 4")
plt.errorbar(t7, D7, label="Манноза")
plt.errorbar(t8, D8, label="Ксилоза, 3")
plt.legend()
plt.show()


plt.title('$D(t)$ (ксилоза)', fontsize=14)
plt.ylabel("$D$, 10$^{-3}$", fontsize=12)
plt.xlabel("$t$, мин", fontsize=12)
plt.grid(True, linestyle="--")
plt.errorbar(t3, D3 * 10**3, label="1")
t3_sample = np.linspace(t3[0], t3[len(t3) - 1], 1000)
plt.errorbar(t4, D4 * 10**3, label="2")
t4_sample = np.linspace(t4[0], t4[len(t4) - 1], 1000)
plt.errorbar(t8, D8 * 10**3, label="3")
t8_sample = np.linspace(t8[0], t8[len(t8) - 1], 1000)
plt.plot(t3_sample, linear(t3_sample, k3 * 10**3, b3 * 10**3),
         label='$D(t) = \\alpha_1 t + d_1$ \n'
               '$\\alpha_1 = (%.1f \\pm %.1f) \cdot $ $\\frac{10^{-6}}{\\text{мин}}$ \n'
               '$d_1 = (%.1f \\pm %.1f) \cdot$ 10$^{-3}$  \n'
               % (k3 * 10**6, sigma_k3 * 10**6, b3 * 10**3, sigma_b3 * 10**3))
plt.plot(t4_sample, linear(t4_sample, k4 * 10**3, b4 * 10**3),
         label='$D(t) = \\alpha_2 t + d_2$ \n'
               '$\\alpha_2 = (%.1f \\pm %.1f) \cdot $ $\\frac{10^{-6}}{\\text{мин}}$ \n'
               '$d_2 = (%.1f \\pm %.1f) \cdot$ 10$^{-3}$ \n'
               % (k4 * 10**6, sigma_k4 * 10**6, b4 * 10**3, sigma_b4 * 10**3))
plt.plot(t8_sample, linear(t8_sample, k8 * 10**3, b8 * 10**3),
         label='$D(t) = \\alpha_3 t + d_3$ \n'
               '$\\alpha_3 = (%.1f \\pm %.1f) \cdot $ $\\frac{10^{-6}}{\\text{мин}}$ \n'
               '$d_3 = (%.1f \\pm %.1f) \cdot$ 10$^{-3}$ \n'
               % (k8 * 10**6, sigma_k8 * 10**6, b8 * 10**3, sigma_b8 * 10**3))
plt.legend()
plt.show()


print('Ксилоза, 1:', '\n',
      'alpha_1 = ', k3, '\pm', sigma_k3, '\n',
      'd1 = ', b3, '\pm', sigma_b3, '\n')

print('Ксилоза, 2:', '\n',
      'alpha_2 = ', k4, '\pm', sigma_k4, '\n',
      'd2 = ', b4, '\pm', sigma_b4, '\n')

print('Ксилоза, 3:', '\n',
      'alpha_3 = ', k8, '\pm', sigma_k8, '\n',
      'd3 = ', b8, '\pm', sigma_b8, '\n')


def speed_calc(num, border_value):
    x_inf, sigma_x_inf = init_curve_fit(num, border_value)[3]
    X, sigma_X = init_curve_fit(num, border_value)[1]

    v = x_inf / (np.sqrt(2 * np.pi) * X * eps * l)
    sigma_v = np.sqrt(sigma_x_inf**2 / (np.sqrt(2 * np.pi) * X * eps * l)**2 +
                      x_inf**2 * sigma_X**2 / (np.sqrt(2 * np.pi) * X**2 * eps * l)**2)
    return [v, sigma_v]


[tau_1, sigma_tau_1], [T_1, sigma_T_1], \
[d0_1, sigma_d0_1], [d_inf_1, sigma_d_inf_1] = init_curve_fit(1, 0.4)

[tau_2, sigma_tau_2], [T_2, sigma_T_2], \
[d0_2, sigma_d0_2], [d_inf_2, sigma_d_inf_2] = init_curve_fit(2, 0.4)

[tau_5, sigma_tau_5], [T_5, sigma_T_5], \
[d0_5, sigma_d0_5], [d_inf_5, sigma_d_inf_5] = init_curve_fit(5, 0.4)

[tau_6, sigma_tau_6], [T_6, sigma_T_6], \
[d0_6, sigma_d0_6], [d_inf_6, sigma_d_inf_6] = init_curve_fit(6, 0.5)

[tau_7, sigma_tau_7], [T_7, sigma_T_7], \
[d0_7, sigma_d0_7], [d_inf_7, sigma_d_inf_7] = init_curve_fit(7, 0.5)


plt.title('$\\frac{d[I_3^{-}]}{dt}[t]$ (глюкоза и манноза)', fontsize=14)
plt.ylabel("$\\frac{d[I_3^{-}]}{dt}$, 10$^{-4}$ $\\frac{\\text{M}}{\\text{мин}}$", fontsize=12)
plt.xlabel("$t$, мин", fontsize=12)
plt.grid(True, linestyle="--")
plt.errorbar(init_select(1, 0.4)[0],
             normal_density(init_select(1, 0.4)[0], tau_1, T_1, d_inf_1) / (eps * l) * 10**4, label="Глюкоза, 1")
plt.errorbar(init_select(2, 0.4)[0],
             normal_density(init_select(2, 0.4)[0], tau_2, T_2, d_inf_2) / (eps * l) * 10**4, label="Глюкоза, 2")
plt.errorbar(init_select(5, 0.4)[0],
             normal_density(init_select(5, 0.4)[0], tau_5, T_5, d_inf_5) / (eps * l) * 10**4, label="Глюкоза, 3")
plt.errorbar(init_select(6, 0.5)[0],
             normal_density(init_select(6, 0.5)[0], tau_6, T_6, d_inf_6) / (eps * l) * 10**4, label="Глюкоза, 4")
plt.errorbar(init_select(7, 0.5)[0],
             normal_density(init_select(7, 0.5)[0], tau_7, T_7, d_inf_7) / (eps * l) * 10**4, label="Манноза")
plt.vlines(tau_1, 0, speed_calc(1, 0.4)[0] * 10**4, linestyle='dashed')
plt.vlines(tau_2, 0, speed_calc(2, 0.4)[0] * 10**4, linestyle='dashed')
plt.vlines(tau_5, 0, speed_calc(5, 0.4)[0] * 10**4, linestyle='dashed')
plt.vlines(tau_6, 0, speed_calc(6, 0.5)[0] * 10**4, linestyle='dashed')
plt.vlines(tau_7, 0, speed_calc(7, 0.5)[0] * 10**4, linestyle='dashed')
plt.legend()
plt.show()


print(speed_calc(1, 0.4))
print(speed_calc(2, 0.4))
print(speed_calc(5, 0.4))
print(speed_calc(6, 0.5))
print(speed_calc(7, 0.5))


s1 = (speed_calc(1, 0.4)[0] + speed_calc(2, 0.4)[0] + speed_calc(5, 0.4)[0]) / 3
sigma_s1 = np.sqrt(speed_calc(1, 0.4)[1]**2 + speed_calc(2, 0.4)[1]**2 + speed_calc(5, 0.4)[1]**2) / 3

print(s1, sigma_s1, '\n')


a1, sigma_a1 = s1 * 2 * 10**(-3) / 20, sigma_s1 * 2 * 10**(-3) / 20
a_xyl, sigma_a_xyl = k8 * 2 * 10**(-3) / (20 * eps * l), sigma_k8 * 2 * 10**(-3) / (20 * eps * l)
a2, sigma_a2 = speed_calc(6, 0.5)[0] * 2 * 10**(-3) / 20, speed_calc(6, 0.5)[1] * 2 * 10**(-3) / 20
a_mann, sigma_a_mann = speed_calc(7, 0.5)[0] * 2 * 10**(-3) / 20, speed_calc(7, 0.5)[1] * 2 * 10**(-3) / 20


print(a1, sigma_a1, '\n')
print(a2, sigma_a2)
print(a_mann, sigma_a_mann)
print(a_xyl, sigma_a_xyl, '\n')

r_xyl, sigma_r_xyl = a_xyl / a2, np.sqrt((sigma_a_xyl / a2)**2 + (sigma_a2 * a_xyl / a2**2)**2)
r_mann, sigma_r_mann = a_mann / a2, np.sqrt((sigma_a_mann / a2)**2 + (sigma_a2 * a_mann / a2**2)**2)

print('Относительная активность ксилозы: ', r_xyl, '\pm', sigma_r_xyl, '\n',
      'Относительная активность маннозы: ', r_mann, '\pm', sigma_r_mann, '\n')
