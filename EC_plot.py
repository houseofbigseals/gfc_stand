
import numpy as np
from scipy.optimize  import curve_fit
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


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


# dataset
x1 = 0.74*np.ones(4)
x2 = 0.59*np.ones(4)
x3= 4.0*np.ones(4)
x4 = 1.73*np.ones(4)
x5= 1.0*np.ones(4)
x6 = 0.43*np.ones(4)

y1 = np.array([2.58, 2.59, 2.61, 2.61])
y2 = np.array([2.31, 2.32, 2.35, 2.33])
# y3 = np.array([3.43, 3.56, 3.54, 3.55])
y3 = np.array([4.09, 4.10, 4.12, 4.15])
print(np.mean(y3))
y4 = np.array([3.53, 3.50, 3.49, 3.55])
print(np.mean(y4))
y5 = np.array([2.91, 2.89, 2.93, 2.93])
y6 = np.array([1.94, 1.91, 1.94, 1.92])

xEC = np.append(x1, x2)
xEC = np.append(xEC, x3)
xEC = np.append(xEC, x4)
xEC = np.append(xEC, x5)
xEC = np.append(xEC, x6)
print(xEC)

yEC = np.append(y1, y2)
yEC = np.append(yEC, y3)
yEC = np.append(yEC, y4)
yEC = np.append(yEC, y5)
yEC = np.append(yEC, y6)
print(yEC)


# lets make approximation
def func(x, a, b, c):
    return a * x*x+ b*x + c

# def func(x, a, b):
#     return a * x +  b


popt, r_squared = approximation_with_r2(func, xEC, yEC)
x_approx = np.arange(start=min(xEC), stop=max(xEC), step=(max(xEC) - min(xEC))/100)
y_approx = func(x_approx, *popt)

# here plotting
fig, ax = plt.subplots(figsize=[16, 9])
plt.grid()
ax.scatter(xEC, yEC, marker='o', color='tab:gray')
ax.plot(x_approx, y_approx, '-', color='b')
# ax.invert_yaxis()
ax.set(ylabel='Выходное напряжение ДК, В')
ax.set(xlabel='Электропроводимость раствора, мС/см')
fig.suptitle("Усреднённая зависимость показаний кондуктометрического датчика "
                "\nот электропроводности раствора")
# str_ = 'Средняя полная влагоемкость субстрата: {} г'.format(int(np.mean([tmc2, tmc3, tmc4])))
# ax.annotate(str_,
#             xy=(0.38, 0.012), xycoords='figure fraction',
#             # horizontalalignment='left', verticalalignment='center',
#             fontsize=10)
# plt.xlim([20, 105])
# plt.ylim([-0.2, -2.0])

# ax.annotate('figure fraction',
#             xy=(.025, .975), xycoords='figure fraction',
#             horizontalalignment='left', verticalalignment='top',
#             fontsize=20)

line4 = mlines.Line2D([], [], color='tab:gray', marker='o', label='Cтатические измерения')
line3 = mlines.Line2D([], [], color='b', label='Аппроксимация\n{:.2e}*x^2 + {:.2e}*x + {:.2f} \nR^2 = {:.2f}'
                                               ''.format(*popt, r_squared))
plt.legend(handles=[line4, line3])
plt.savefig('xEC.png')
plt.show()
