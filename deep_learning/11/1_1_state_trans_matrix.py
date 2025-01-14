from matplotlib import pyplot

if __name__ == '__main__':
    pyplot.rcParams['text.usetex'] = True
    matrix_str = r'''$P = \left[ \begin{array}{cccc} P(s_1|s_1) & P(s_2|s_1) & \cdots & P(s_n|s_1) \\ P(s_1|s_2) & P(s_2|s_2) & \cdots & P(s_n|s_2) \\ \cdots & \cdots & \ddots & \cdots \\ P(s_1|s_n) & P(s_2|s_n) & \cdots & P(s_n|s_n) \end{array} \right]$'''
    pyplot.text(0.5, 0.5, matrix_str, fontsize=20, ha='center', va='center')
    pyplot.axis('off')
    pyplot.show()
