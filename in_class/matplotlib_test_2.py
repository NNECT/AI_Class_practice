import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import re


def gdp_show():
    # 기본으로 들어가야함
    # plt.rc('font', family='GULIM')
    # mpl.rcParams['axes.unicode_minus'] = False

    font_name = mpl.font_manager.FontProperties(fname='C:\Windows\Fonts\gulim.ttc').get_name()
    plt.rc('font', family=font_name)

    filename = "2016_GDP.txt"
    f = open(filename, encoding='utf-8')
    data = np.array([re.sub(',', '', line.strip()).split(':') for line in f.readlines()[1:]])
    f.close()

    country = data[:10, 1]
    gdp = np.int32(data[:10, 2])

    column = np.arange(len(gdp))
    # plt.bar(column, gdp, color=['tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:blue'])
    # plt.bar(column, gdp, color=['#091A7A', '#1939B7', '#3366FF', '#84A9FF', '#D6E4FF'])
    # plt.bar(column, gdp, color=mpl.colors.CSS4_COLORS)
    plt.bar(column, gdp, color=mpl.colors.TABLEAU_COLORS)
    plt.xticks(column, country)

    plt.show()


if __name__ == "__main__":
    gdp_show()
