from matplotlib import pyplot

if __name__ == '__main__':
    pyplot.rcParams['text.usetex'] = True
    matrix_str = r'''$P = \left[ \begin{array}{cccccc} 0.9 & 0.1 & 0 & 0 & 0 & 0 \\ 0.5 & 0 & 0.5 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0.6 & 0 & 0.4 \\ 0 & 0 & 0 & 0 & 0.3 & 0.7 \\ 0 & 0.2 & 0.3 & 0.5 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{array} \right]$'''
    pyplot.text(0.5, 0.5, matrix_str, fontsize=20, ha='center', va='center')
    pyplot.axis('off')
    pyplot.show()