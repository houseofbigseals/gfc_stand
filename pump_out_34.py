import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from pump_out_39 import create_water_array, approximation_with_r2


module_num = 34


def calculate_tmc(full_mass):
    # lets calculate total moisture capacity of root module
    # full_module_masses = [900, 889, 890]  # g

    dm_soil_substitute = 63  # g
    dm_iron_frame = 230  # g
    dm_water_in_porous_tube = 16.5405  # cm3 ~ g of water
    dm_water_in_perforated_tube = 5.1051  # cm3 ~ g of water
    tmc = full_mass - dm_soil_substitute - \
                              dm_iron_frame - dm_water_in_porous_tube - dm_water_in_perforated_tube
    print("total moisture capacity  = {}".format(tmc))
    return tmc


# arrays creation

# 1
# 17:52, 15/июля/20 трубка 34, масса намоченной трубки 900 г
y1 = np.array([-0.48, -1.11, -0.85, -1.32, -0.92, -1.6, -1.03, -1.6, -1.14, -1.63, -1.22, -1.83, -1.3, -1.9,
      -1.4, -2.03, -1.45, -1.9, -1.57, -2.19, -1.6, -1.85, -1.8, -2.4, -1.92, -3.2])
#x1 = create_water_array(y1, 900)
tmc1 = calculate_tmc(900)
x1 = create_water_array(y1, tmc1)/(tmc1*0.01)
cluster1 = np.array([i % 2 for i in range(0, len(y1))])

# 5
# 27/07/2020 - Трубка 34 916 гр, откачка - 889 гр.
y5 = np.array([-0.65, -1.42, -0.82, -1.05, -0.92, -1.55, -0.72, -1.42, -1.01, -1.8, -1.05, -2.4,
               -1.1, -1.45, -1.2, -2.15, -1.35, -1.98, -1.32, -2.15, -1.45, -2.3, -1.55, -2.15,
               -1.56, -2.7, -1.7, -3.35, -1.9])
# x5 = create_water_array(y5, 889)
tmc5 = calculate_tmc(889)
x5 = create_water_array(y5, tmc5)/(tmc5*0.01)
print(x5)
cluster5 = np.array([i % 2 for i in range(0, len(y5))])

# 6
# 3/08/2020 - Трубка 34 Вес: 940, 890 откачка.
y6 = np.array([
    -0.33, -1.81, -0.51, -1.85, -0.72, -1.84, -0.82, -2.15, -0.95, -2.17, -1.07, -2.5,
    -1.17, -2.5, -1.25, -2.05, -1.4, -2.4, -1.8, -2.17, -1.4, -2.45, -1.68, -2.6, -1.8, -2.6,
    -1.92, -3.01, -2.1
])
x_fake = np.array([10, 15, 20])
y_fake = np.array([-3.0, -3.0, -3.0])

# x6 = create_water_array(y6, 890)
tmc6 = calculate_tmc(890)
x6 = create_water_array(y6, tmc6)/(tmc6*0.01)
# x6 = create_water_array(y6, tmc)

print(x6)
cluster6 = np.array([i % 2 for i in range(0, len(y6))])

y34_array = np.append(y1[cluster1==0][1:], y5[cluster5==0][1:-1])
y34_array = np.append(y34_array, y6[cluster6==0][1:])
x34_array = np.append(x1[cluster1==0][1:], x5[cluster5==0][1:-1])
x34_array = np.append(x34_array, x6[cluster6==0][1:])
print("++++++++++++++++++++++++")
print(y6[cluster6==0][0:])
print("++++++++++++++++++++++++")
print(x6[cluster6==0][0:])
print("++++++++++++++++++++++++")
print(create_water_array(y6, tmc6)[cluster6==0][0:])
print("++++++++++++++++++++++++")

print("x34_array len is: ",len(x34_array))
print(x34_array[-1])

x_train = np.append(x34_array, x_fake)
y_train = np.append(y34_array, y_fake)

# here interpolation
# using this https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
# pol = np.polyfit(x34_array, y34_array, 3)  # array with polynom coeffs
# print(pol)
# p_obj = np.poly1d(pol)  # smart object to calculate polynom values for this coeffs
x_approx = np.arange(start=x34_array[-1], stop=x34_array[0], step=(-x34_array[-1] + x34_array[0])/100)

# x_approx = x_train

print((-x34_array[-1] + x34_array[0])/100)
print(x_approx)
# y_approx = p_obj(x_approx)

def func(x, a, b, c, d):
    return a*np.power(x, 3) + b* np.power(x, 2) + c*x + d

# def func(x, a, b, c, d, e, f):
#     return a*np.power(x, 5) + b*np.power(x, 4) + c*np.power(x, 3) + d*np.power(x, 2) + e*x + f

# popt, r_squared = approximation_with_r2(func, x34_array, y34_array)
popt, r_squared = approximation_with_r2(func, x_train, y_train)
# popt2 = [4.11676807e+01, -7.21362717e-05,  2.18977786e-02, -2.50286343e+00]
y_approx = func(x_approx, *popt)

print("x34_array len is: ",len(x34_array))

# here plotting
fig, ax = plt.subplots(figsize=[16, 9])
plt.grid()
ax.scatter(x34_array, y34_array, marker='o', color='tab:gray')
ax.plot(x_approx, y_approx, '-', color='b')
ax.invert_yaxis()
str_ = 'Средняя полная влагоемкость субстрата: {} г'.format(int(np.mean([tmc1, tmc5, tmc6])))
ax.annotate(str_,
            xy=(0.38, 0.012), xycoords='figure fraction',
            # horizontalalignment='left', verticalalignment='center',
            fontsize=10)

ax.set(ylabel='Давление в пористой трубке КМ, кПа')
ax.set(xlabel='Доля от полной влагоемкости cубстрата в КМ, %')
fig.suptitle("ОГХ корневого модуля {} \nветвь осушения".format(module_num))

line4 = mlines.Line2D([], [], color='tab:gray', marker='o', label='Cтатические измерения')
line3 = mlines.Line2D([], [], color='b', label='Аппроксимация\n'
                                               '{:.2e}*x^3 {:.2e}*x^2+{:.2e}*x {:.2f} \nR^2 = {:.2f}'
                                               ''.format(*popt, r_squared))
plt.xlim([20, 100])
plt.ylim([-0.25, -2.5])

plt.legend(handles=[line4, line3])
plt.savefig('x{}.png'.format(module_num))
plt.show()
