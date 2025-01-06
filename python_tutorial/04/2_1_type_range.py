
for i in range(5, 10):
    print(i)
    
assert list(range(10)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
assert list(range(1, 11)) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
assert list(range(0, 30, 5)) == [0, 5, 10, 15, 20, 25]
assert list(range(0, 10, 3)) == [0, 3, 6, 9]
assert list(range(0, -10, -1)) == [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
assert list(range(0)) == []
assert list(range(1, 0)) == []
