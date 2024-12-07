import numpy
import pandas

file_path = 'temp/personal_info.xlsx'

if __name__ == '__main__':
    arr = numpy.loadtxt('res/deep_learning/simple_int.csv',
                        delimiter=',', skiprows=1)
    print(arr)

    df = pandas.DataFrame({
        "Number": numpy.array([1, 2, 3]),
        "Age": numpy.array([25, 30, 22])
    })
    df.to_excel(file_path, index=False)
    
    df = pandas.read_excel(file_path)
    data = df.to_numpy()
    print(data)
