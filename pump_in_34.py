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

# # 1
# # 17:52, 15/июля/20 трубка 34, масса намоченной трубки 900 г
# y1 = np.array([-0.48, -1.11, -0.85, -1.32, -0.92, -1.6, -1.03, -1.6, -1.14, -1.63, -1.22, -1.83, -1.3, -1.9,
#       -1.4, -2.03, -1.45, -1.9, -1.57, -2.19, -1.6, -1.85, -1.8, -2.4, -1.92, -3.2])
# #x1 = create_water_array(y1, 900)
# tmc1 = calculate_tmc(900)
# x1 = create_water_array(y1, tmc1)/(tmc1*0.01)
# cluster1 = np.array([i % 2 for i in range(0, len(y1))])

# 8
# 34 13.08.2021
# y8 = np.array([-2.15, -1.05, -0.89, -0.70, -0.52, -0.45, -0.48, -0.35])
y8 = np.array([-2.15, -1.05, -0.89, -0.70, -0.45, -0.40,-0.48, -0.40])
x8 = []
x8_deltas = np.array([50, 30, 30, 30, 20, 30, 30, 30])
tmc8 = calculate_tmc(532)
last_portion = tmc8

# y_fake = [0.40]
# x_fake = [30]

# x8_deltas = np.append()

for dx in x8_deltas:
    x8.append(last_portion)
    last_portion = last_portion + dx

x8 = np.array(x8)
# tmc1 = calculate_tmc(900)


y34_array = y8
x34_array = x8/(0.01*578)
print("++++++++++++++++++++++++")
print(x34_array)
print("++++++++++++++++++++++++")
print(y34_array)
print("++++++++++++++++++++++++")

print("x34_array[0]",x34_array[0])
print(x34_array[-1])

# here interpolation
# using this https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
pol = np.polyfit(x34_array, y34_array, 3)  # array with polynom coeffs
print(pol)
p_obj = np.poly1d(pol)  # smart object to calculate polynom values for this coeffs
x_approx = np.arange(start=x34_array[-1], stop=x34_array[0], step=(-x34_array[-1] + x34_array[0])/100)
print((-x34_array[-1] + x34_array[0])/100)
print(x_approx)
# y_approx = p_obj(x_approx)

def func(x, a, b, c, d):
    return a*np.power(x, 3) + b* np.power(x, 2) + c*x + d

# def func(x, a, b, c):
#     return a*np.power(x, 2) + b* x + c

popt, r_squared = approximation_with_r2(func, x34_array, y34_array)

y_approx = func(x_approx, *popt)

# here plotting
fig, ax = plt.subplots(figsize=[12, 9])
plt.grid()
ax.scatter(x34_array, y34_array, marker='o', color='tab:gray')
ax.plot(x_approx, y_approx, '-', color='b')
ax.invert_yaxis()
str_ = 'Средняя полная влагоемкость cубстрата: {} г'.format(578)
ax.annotate(str_,
            xy=(0.38, 0.012), xycoords='figure fraction',
            # horizontalalignment='left', verticalalignment='center',
            fontsize=10)



ax.set(ylabel='Давление в пористой трубке КМ, кПа')
ax.set(xlabel='Доля от полной влагоемкости cубстрата в КМ, %')
fig.suptitle("ОГХ корневого модуля {} \nветвь увлажнения".format(module_num))

line4 = mlines.Line2D([], [], color='tab:gray', marker='o', label='Cтатические измерения')
line3 = mlines.Line2D([], [], color='b', label='Аппроксимация\n'
                                               # '{:.2e}*x^3 {:.2e}*x^2+{:.2e}*x {:.2f} \nR^2 = {:.2f}'
                                               '{:.2e}*x^3 {:.2e}*x^2+{:.2e}*x {:.2f} \nR^2 = {:.2f}'
                                               ''.format(*popt, r_squared))

plt.xlim([20, 100])
plt.ylim([-0.25, -2.2])

plt.legend(handles=[line4, line3])
plt.savefig('x34_in.png')
plt.show()
