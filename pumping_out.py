

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines


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
    dm_iron_frame = 230  # g
    dm_water_in_porous_tube = 16.5405  # cm3 ~ g of water
    dm_water_in_perforated_tube = 5.1051  # cm3 ~ g of water
    tmc = full_mass - dm_soil_substitute - \
                              dm_iron_frame - dm_water_in_porous_tube - dm_water_in_perforated_tube
    print("total moisture capacity  = {}".format(tmc))
    return tmc


if __name__ == "__main__":

    # 1
    # 17:52, 15/июля/20 трубка 34, масса намоченной трубки 900 г
    y1 = np.array([-0.48, -1.11, -0.85, -1.32, -0.92, -1.6, -1.03, -1.6, -1.14, -1.63, -1.22, -1.83, -1.3, -1.9,
          -1.4, -2.03, -1.45, -1.9, -1.57, -2.19, -1.6, -1.85, -1.8, -2.4, -1.92, -3.2])
    # x1 = create_water_array(y1, 900)
    tmc1 = calculate_tmc(900)
    x1 = create_water_array(y1, tmc1) / (tmc1 * 0.01)
    cluster1 = np.array([i % 2 for i in range(0, len(y1))])

    # 2
    # 17:07, 16/июля/20 трубка 39, время замачивания 42 часа (при это посередине этого времени его вынимали и немного откачали)
    y2 = np.array([
        -0.3, -0.7, -0.6, -1.01, -0.78, -1.05, -0.9, -1.24, -1.02, -1.14, -1.05, -1.35, -1.14, -1.42,
        -1.2, -1.4, -1.3, -1.6, -1.34, -1.8, -1.35, -1.7, -1.5, -1.8, -1.56, -1.85, -1.7, -1.99
    ])
    tmc2 = calculate_tmc(900)
    x2 = create_water_array(y2, tmc2) / (tmc2 * 0.01)  ## value 900 is not real, it is 900 +- 20 g
    cluster2 = np.array([i % 2 for i in range(0, len(y2))])

    # 3
    # 19.07.2021 трубка 39 Осушение, 15:35, масса намоченной трубки 924 г, замачивание около 60 ч, Ратм.=98,87

    y3 = np.array([
        -0.17, -0.54, -0.4, -0.7, -0.56, -0.84, -0.72, -0.95, -0.88, -1.04, -0.98, -1.2,
        -0.96, -1.17, -1.14, -1.26, -1.22, -1.32, -1.22, -1.37, -1.33, -1.44, -1.35, -1.59,
        -1.46, -1.62, -1.57, -1.77, -1.68, -1.9, -1.82
    ])

    tmc3 = calculate_tmc(924)
    x3 = create_water_array(y3, tmc3) / (tmc3 * 0.01)
    cluster3 = np.array([i % 2 for i in range(0, len(y3))])


    # 4
    # 21.07.2021 трубка номер 39 Осушение время 16:00,  масса трубки 884г, время замачивания 20 часов ,
    # атм давление 98.93кПа
    date_ = "21.07.2021"
    module_num = "39"
    start_x4 = 884  # g
    atm_pressure = "98.93 kPa"
    soaking_time = "20 hrs"

    y4 = np.array([-0.47, -0.52, -0.39, -0.73, -0.61, -0.87, -0.78, -0.99, -0.9, -1.2, -1, -1.14,
         -1.09, -1.2, -1.18, -1.29, -1.28, -1.5, -1.3, -1.6, -1.35, -1.48, -1.47, -1.57,
         -1.57, -1.68, -1.62, -1.82, -1.71, -2])  # kPa, data from pressure sensor

    tmc4 = calculate_tmc(884)
    x4 = create_water_array(y4, tmc4) / (tmc4 * 0.01)

    print(x4)
    cluster4 = np.array([i % 2 for i in range(0, len(y4))])


    # 5
    # 27/07/2020 - Трубка 34 916 гр, откачка - 889 гр.
    y5 = np.array([-0.65, -1.42, -0.82, -1.05, -0.92, -1.55, -0.72, -1.42, -1.01, -1.8, -1.05, -2.4,
          -1.1, -1.45, -1.2, -2.15, -1.35, -1.98, -1.32, -2.15, -1.45, -2.3, -1.55, -2.15,
          -1.56, -2.7, -1.7, -3.35, -1.9])
    tmc5 = calculate_tmc(889)
    x5 = create_water_array(y5, tmc5) / (tmc5 * 0.01)

    print(x5)
    cluster5 = np.array([i % 2 for i in range(0, len(y5))])

    # 6
    # 3/08/2020 - Трубка 34 Вес: 940, 890 откачка.
    y6 = np.array([
        -0.33, -1.81, -0.51, -1.85, -0.72, -1.84, -0.82, -2.15, -0.95, -2.17, -1.07, -2.5,
        -1.17, -2.5, -1.25, -2.05, -1.4, -2.4, -1.8, -2.17, -1.4, -2.45, -1.68, -2.6, -1.8, -2.6,
        -1.92, -3.01, -2.1
    ])

    tmc6 = calculate_tmc(890)
    x6 = create_water_array(y6, tmc6) / (tmc6 * 0.01)

    print(x6)
    cluster6 = np.array([i % 2 for i in range(0, len(y6))])

    # pumping in
    # 7
    # 34 11.08.2021
    y7 = np.array([-3.7, -1.97, -1.74, -1.7, -1.65, -1.61])

    x7 = []
    x7_deltas = np.array([81, 73, 65, 56, 48.6, 40.5])
    last_portion = 541

    for dx in x7_deltas:
        x7.append(last_portion)
        last_portion = last_portion + dx

    x7 = np.array(x7)

    # 8
    # 34 13.08.2021
    y8 = np.array([-2.15, -1.05, -0.89, -0.70, -0.52, -0.45, -0.48, -0.35])
    x8 = []
    x8_deltas = np.array([50, 30, 30, 30, 20, 30, 30, 30])
    last_portion = 532

    for dx in x8_deltas:
        x8.append(last_portion)
        last_portion = last_portion + dx

    # 9
    # 39 22.07.2021
    y9 = np.array([-2.1, -0.83, -0.45, -0.36, -0.33, -0.4, -0.36, -0.3, -0.25, -0.2])
    x9 = []
    x9_deltas = np.array([31.32, 31.32,31.32,31.32,31.32,31.32,31.32,31.32,31.32,31.32])
    last_portion = 419

    for dx in x9_deltas:
        x9.append(last_portion)
        last_portion = last_portion + dx


    # 10
    # 39 12.08.2021
    y10 = np.array([-3.17, -2.80, -2.6, -2.6, -2.4, -2.25])
    x10 = []
    x10_deltas = np.array([73, 65, 56.7, 48.8, 40.5, 32.4])
    last_portion = 550

    for dx in x10_deltas:
        x10.append(last_portion)
        last_portion = last_portion + dx


    # plot final graph

    fig, ax = plt.subplots(figsize=[16, 9])
    plt.grid()

    # -- 39 : 2, 3, 4

    # plt.quiver(x4[:-1], y4[:-1], x4[1:]-x4[:-1], y4[1:]-y4[:-1], scale_units='xy', angles='xy', scale=1, width=0.0025, color='c')
    # plt.quiver(x2[:-1], y2[:-1], x2[1:] - x2[:-1], y2[1:] - y2[:-1], scale_units='xy', angles='xy', scale=1,
    #             width=0.0025,
    #             color='c')
    # plt.quiver(x3[:-1], y3[:-1], x3[1:] - x3[:-1], y3[1:] - y3[:-1], scale_units='xy', angles='xy', scale=1,
    #             width=0.0025,
    #             color='c')

    # ax.plot(x4[cluster4==0],y4[cluster4==0], marker='s', color='c')
    # ax.scatter(x4[cluster4 == 1], y4[cluster4 == 1], marker='^', color='b')
    # ax.scatter(x4[cluster4==0],y4[cluster4==0], marker='s', color='b')
    #
    #
    # ax.plot(x2[cluster2 == 0], y2[cluster2 == 0], marker='s', color='c')
    # ax.scatter(x2[cluster2 == 1], y2[cluster2 == 1], marker='^', color='b')
    # ax.scatter(x2[cluster2==0],y2[cluster2==0], marker='s', color='b')
    # #
    #
    # ax.plot(x3[cluster3 == 0], y3[cluster3 == 0], marker='s', color='c')
    # ax.scatter(x3[cluster3 == 1], y3[cluster3 == 1], marker='^', color='b')
    # ax.scatter(x3[cluster3==0],y3[cluster3==0], marker='s', color='b')
    ax.invert_yaxis()
    # -- 34 : 1, 5, 6

    # plt.quiver(x[:0:-1], y[:0:-1], -1*(x[:0:-1]-x[-2::-1]), -1*(y[:0:-1]-y[-2::-1]), scale_units='xy', angles='xy', scale=1)
    plt.quiver(x1[:-1], y1[:-1], x1[1:] - x1[:-1], y1[1:] - y1[:-1], scale_units='xy', angles='xy', scale=1, width=0.0025,
               color='m')
    #
    plt.quiver(x5[:-1], y5[:-1], x5[1:] - x5[:-1], y5[1:] - y5[:-1], scale_units='xy', angles='xy', scale=1, width=0.0025,
               color='m')
    #
    plt.quiver(x6[:-1], y6[:-1], x6[1:] - x6[:-1], y6[1:] - y6[:-1], scale_units='xy', angles='xy', scale=1, width=0.0025,
               color='m')
    #
    ax.scatter(x1[cluster1==1],y1[cluster1==1], marker='^', color='g')
    # ax.plot(x1[cluster1 == 0], y1[cluster1 == 0], marker='s', color='m')
    ax.scatter(x1[cluster1==0],y1[cluster1==0], marker='s', color='g')
    #
    ax.scatter(x5[cluster5==1],y5[cluster5==1], marker='^', color='g')
    ax.scatter(x5[cluster5==0],y5[cluster5==0], marker='s', color='g')
    # ax.plot(x5[cluster5 == 0], y5[cluster5 == 0], marker='s', color='m')
    #
    ax.scatter(x6[cluster6==1],y6[cluster6==1], marker='^', color='g')
    ax.scatter(x6[cluster6==0],y6[cluster6==0], marker='s', color='g')
    # ax.plot(x6[cluster6 == 0], y6[cluster6 == 0], marker='s', color='m')

    # pumping in plots
    # ax.plot(x7, y7, marker='x', color='g')
    # ax.plot(x8, y8, marker='x', color='g')
    # ax.plot(x9, y9, marker='x', color='b')
    # ax.plot(x10, y10, marker='x', color='b')

    module_num = 34

    ax.set(ylabel='Давление в пористой трубке КМ, кПа')
    ax.set(xlabel='Доля от полной влагоемкости cубстрата в КМ, %')
    fig.suptitle("Динамика изменения "
                 "давления в пористой трубке корневого модуля {} \n"
                 "в режиме осушения путём откачки доз воды из пористой трубки".format(module_num))

    line4 = mlines.Line2D([], [], color='tab:green', marker='^', label='Давление сразу после \nоткачки '
                                                                       'очередных доз  воды')
    # line3 = mlines.Line2D([], [], color='b', label='Модуль 34, осушение - аппроксимация')
    line1 = mlines.Line2D([], [], color='tab:green', marker='s', label='Давление через 60 мин после\nоткачки '
                                                                       'очередных доз воды')
    # line2 = mlines.Line2D([], [], color='c', label='Модуль 34, увлажнение - аппроксимация')
    plt.legend(handles=[line4, line1])

    # ax.set(ylabel='Pressure in module, kPa')
    # ax.set(xlabel='Water mass in module, g')
    # fig.suptitle("date={}, module_id={}, start_mass={}g, atm_pressure={}, soaking_time={} ".format(
    #     date_,
    #     module_num,
    #     start_xn,
    #     atm_pressure,
    #     soaking_time
    #     ))
    # fig.suptitle("General hydrophysical characteristic 2021")

    # triang = mlines.Line2D([], [], color='blue', marker='^',
    #                           markersize=7, label='After pumping')
    # square = mlines.Line2D([], [], color='blue', marker='s',
    #                           markersize=7, label='Before pumping')
    # line1 = mlines.Line2D([], [], color='c', marker='s', label='Module 39 pumping out')
    # line2 = mlines.Line2D([], [], color='m', marker='s', label='Module 34 pumping out')
    # line3 = mlines.Line2D([], [], color='b', marker='x', label='Module 39 pumping in')
    # line4 = mlines.Line2D([], [], color='g', marker='x', label='Module 34 pumping in')

    # plt.legend(handles=[ line1, line2, line3, line4])
    plt.savefig('x_{}_out_saw.png'.format(module_num))
    plt.show()

