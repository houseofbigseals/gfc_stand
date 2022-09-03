import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
module_num = 39


def approximation_with_r2(func, x, y):
    popt, pcov = curve_fit(func, x, y)
    print("popt using scipy: {}".format(popt))
    print("pcov using scipy: {}".format(pcov))
    # perr = np.sqrt(np.diag(pcov))
    # print("perr using scipy: {}".format(perr))

    # to compute R2
    # https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit

    residuals = y - func(x, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print("r_squared using custom code: {}".format(r_squared))
    return popt, r_squared


def create_water_array(y, start_x):
    x_deltas = [29.97 * ((i + 1) % 2) for i in range(0, len(y))]  # sometimes delta is 0, sometimes is 29.97
    print(x_deltas)

    x = []
    last_portion = start_x

    for dx in x_deltas:
        x.append(last_portion)
        last_portion = last_portion - dx

    x = np.array(x)  # final array with mass of water in roots module
    return x

def calculate_tmc(full_mass):
    # lets calculate total moisture capacity of root module
    # full_module_masses = [900, 889, 890]  # g

    dm_soil_substitute = 63  # g
    dm_iron_frame = 224.4  # g
    dm_water_in_porous_tube = 16.5405  # cm3 ~ g of water
    dm_water_in_perforated_tube = 5.1051  # cm3 ~ g of water
    tmc = full_mass - dm_soil_substitute - \
                              dm_iron_frame - dm_water_in_porous_tube - dm_water_in_perforated_tube
    print("total moisture capacity  = {}".format(tmc))
    return tmc

if __name__ == "__main__":
    # arrays creation


    # 2
    # 17:07, 16/июля/20 трубка 39, время замачивания 42 часа (при это посередине этого времени его вынимали и немного откачали)
    y2 = np.array([
        -0.3, -0.7, -0.6, -1.01, -0.78, -1.05, -0.9, -1.24, -1.02, -1.14, -1.05, -1.35, -1.14, -1.42,
        -1.2, -1.4, -1.3, -1.6, -1.34, -1.8, -1.35, -1.7, -1.5, -1.8, -1.56, -1.85, -1.7, -1.99
    ])
    # x2 = create_water_array(y2, 900) ## value 900 is not real, it is 900 +- 20 g
    tmc2 = calculate_tmc(900)
    x2 = create_water_array(y2, tmc2)/(tmc2*0.01)
    cluster2 = np.array([i % 2 for i in range(0, len(y2))])

    # 3
    # 19.07.2021 трубка 39 Осушение, 15:35, масса намоченной трубки 924 г, замачивание около 60 ч, Ратм.=98,87

    y3 = np.array([
        -0.17, -0.54, -0.4, -0.7, -0.56, -0.84, -0.72, -0.95, -0.88, -1.04, -0.98, -1.2,
        -0.96, -1.17, -1.14, -1.26, -1.22, -1.32, -1.22, -1.37, -1.33, -1.44, -1.35, -1.59,
        -1.46, -1.62, -1.57, -1.77, -1.68, -1.9, -1.82
    ])

    # x3 = create_water_array(y3, 924)
    tmc3 = calculate_tmc(924)
    x3 = create_water_array(y3, tmc3)/(tmc3*0.01)
    cluster3 = np.array([i % 2 for i in range(0, len(y3))])


    # 4
    # 21.07.2021 трубка номер 39 Осушение время 16:00,  масса трубки 884г, время замачивания 20 часов ,
    # атм давление 98.93кПа
    date_ = "21.07.2021"
    #module_num = "39"
    start_x4 = 884  # g
    atm_pressure = "98.93 kPa"
    soaking_time = "20 hrs"

    y4 = np.array([-0.47, -0.52, -0.39, -0.73, -0.61, -0.87, -0.78, -0.99, -0.9, -1.2, -1, -1.14,
         -1.09, -1.2, -1.18, -1.29, -1.28, -1.5, -1.3, -1.6, -1.35, -1.48, -1.47, -1.57,
         -1.57, -1.68, -1.62, -1.82, -1.71, -2])  # kPa, data from pressure sensor

    # x4 = create_water_array(y4, start_x4)
    tmc4 = calculate_tmc(884)
    x4 = create_water_array(y4, tmc4)/(tmc4*0.01)

    print(x4)
    cluster4 = np.array([i % 2 for i in range(0, len(y4))])


    y39_array = np.append(y2[cluster2==0][1:], y3[cluster3==0][1:])
    y39_array = np.append(y39_array , y4[cluster4==0][1:])
    x39_array = np.append(x2[cluster2==0][1:], x3[cluster3==0][1:])
    x39_array = np.append(x39_array , x4[cluster4==0][1:])
    print("++++++++++++++++++++++++")
    print(x4[cluster4==0])
    print("++++++++++++++++++++++++")
    print(y4[cluster4==0])
    print("++++++++++++++++++++++++")
    print(create_water_array(y4, tmc4)[cluster4==0])
    print("++++++++++++++++++++++++")
    print(np.shape(x39_array[-2]))
    print(np.shape(y39_array))
    print("x39_array[-1]", x39_array[-1])

    # here interpolation
    # using this https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html


    res = np.polyfit(x39_array, y39_array, 3, full=True)  # array with polynom coeffs
    pol = res[0]
    print("polynom", pol)
    # print(residuals, rank, singular_values, rcond)
    print("residuals and other",res[1:])
    # lets calculate R2
    # results = {}
    #
    # coeffs = numpy.polyfit(x, y, degree)
    #
    # # Polynomial Coefficients
    # results['polynomial'] = coeffs.tolist()
    #
    # # r-squared
    # p = np.poly1d(coeffs)
    # # fit values, and mean
    # yhat = p(x)  # or [p(z) for z in x]
    # ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    # ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    # sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    # results['determination'] = ssreg / sstot


    p_obj = np.poly1d(pol)  # smart object to calculate polynom values for this coeffs
    x_approx = np.arange(start=x39_array[-1], stop=x39_array[0], step=(-x39_array[-1] + x39_array[0])/100)
    # print((-x39_array[-1] + x39_array[0])/100)
    # print(x_approx)
    y_approx = p_obj(x_approx)
    print("fuuuuuuuuu")




    # another approximation using
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    def func(x, a, b, c, d):
        return a*np.power(x, 3) + b* np.power(x, 2) + c*x + d

    popt, r_squared = approximation_with_r2(func, x39_array, y39_array)
    print("less shhhh: {}".format(func(np.array([6, 8, 10, 12, 14, 16, 18, 20]), *popt)))
    # here plotting
    fig, ax = plt.subplots(figsize=[16, 9])
    plt.grid()
    ax.scatter(x39_array, y39_array, marker='o', color='tab:gray')
    ax.plot(x_approx, func(x_approx, *popt), '-', color='b')
    ax.invert_yaxis()
    ax.set(ylabel='Давление в пористой трубке КМ, кПа')
    ax.set(xlabel='Доля от полной влагоемкости cубстрата в КМ, %')
    fig.suptitle("ОГХ корневого модуля {} \nветвь осушения".format(module_num))
    str_ = 'Средняя полная влагоемкость субстрата: {} г'.format(int(np.mean([tmc2, tmc3, tmc4])))
    ax.annotate(str_,
                xy=(0.38, 0.012), xycoords='figure fraction',
                # horizontalalignment='left', verticalalignment='center',
                fontsize=10)
    plt.xlim([20, 105])
    plt.ylim([-0.2, -2.0])

    # ax.annotate('figure fraction',
    #             xy=(.025, .975), xycoords='figure fraction',
    #             horizontalalignment='left', verticalalignment='top',
    #             fontsize=20)

    line4 = mlines.Line2D([], [], color='tab:gray', marker='o', label='Cтатические измерения')
    line3 = mlines.Line2D([], [], color='b', label='Аппроксимация\n{:.2e}*x^3 {:.2e}*x^2+{:.2e}*x {:.2f} \nR^2 = {:.2f}'
                                                   ''.format(*pol, r_squared))
    plt.legend(handles=[line4, line3])
    plt.savefig('x{}.png'.format(module_num))
    plt.show()
