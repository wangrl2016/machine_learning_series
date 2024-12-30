import numpy

if __name__ == '__main__':
    a = numpy.array([('Rex', 9, 81), ('Fido', 3, 27.0)],
                     dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
    print(a)
    tuple(a[1]) == ('Fido', 3, 27.0)
    assert (a['age'] == [9, 3]).all()
    a['age'] = 5
    print(a)
