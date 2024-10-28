if __name__ == '__main__':
    # integer
    a = 10
    b = 0o25
    c = 0x1F
    print(a, end=' ')
    print(b, end=' ')
    print(c, end=' ')
    # float
    d = 3.14
    e = 5.67e-2
    print(d, end=' ')
    print(e, end=' ')
    # complex
    f = 3+4j
    print(f)

    g = int('123')
    print(g, end=' ')
    i = float(3.14)
    print(i, end=' ')
    j = complex(1, 3)
    print(j)

    # get type of common type
    print(type(1))
    print(type('Hello'))
    print(type(True))
    print(type([1, 2, 3]))
    print(type((1, 2, 3)))
    print(type({'a': 1, 'b': 2}))
    print(type(set([1, 2, 3])))

    print("Hello, world!")
    print('Got your back')
    # use \' to escape the single quote
    print('doesn\'t')
    print("doesn't")
    print('First line.\nSecond line.')

    print(r'C:\some\name')

    print(3 * 'un' + 'ium')
    print('Py' 'thon')
    text = ('Put several strings within parentheses '
            'to have them joined together.')
    print(text)

    word = 'Python'
    print(word[0], end=' ')
    print(word[5], end=' ')
    print(word[-1], end=' ')
    print(word[-2], end=' ')
    print(word[-6], end=' ')
    print(word[0:2], end=' ')    # from the beginning to position 2 (excluded)
    print(word[2:5], end=' ')    # from position 2 (included) to 5 (excluded)
    print(word[:2], end=' ')     # from the beginning to position 2 (excluded)
    print(word[4:], end=' ')     # from position 4 (included) to the end
    print(word[-2:], end=' ')    # from the second-last (included) to the end
    print(word[:2] + word[2:], end=' ')
    print(word[:4] + word[4:], end=' ')
    print(word[4:42], end=' ')
    print('J' + word[1:], end=' ')
    print(word[:2] + 'py')

    squares = [1, 4, 9, 16, 25]
    print(squares)
    print(squares[0], end=' ')   # indexing returns a value
    print(squares[-1])
    print(squares[-3:]) # slicing returns a new list
    print(squares + [36, 49, 64, 81, 100])

    cubes = [1, 8, 27, 65, 125]
    print(cubes)
    cubes[3] = 64   # replace the wrong value
    print(cubes)

    cubes.append(216)   # add the cube of 6
    cubes.append(7 ** 3)    # and the cube of 7
    print(cubes)

    rgb = ['Red', 'Green', 'Blue']
    rgba = rgb
    print('Is the address of two objects the same?', id(rgb) == id(rgba))
    rgba.append('Alph')
    print(rgb)

    correct_rgba = rgba[:]
    correct_rgba[-1] = 'Alpha'
    print(correct_rgba)
    print(rgba)
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    print(letters)
    letters[2:5] = ['C', 'D', 'E']  # replace some values
    print(letters)
    letters[2:5] = []  # remove some values
    print(letters)
    letters[:] = []  # clear the list by replacing all the elements with an empty list
    print(letters)

    a = ['a', 'b', 'c']
    b = [1, 2, 3]
    nest_list = [a, b]
    print(nest_list)
    print(nest_list[0])
    print(nest_list[0][1])

    # Fibonacci series:
    # the sum of two elements defines the next
    a, b = 0, 1
    while a < 1000:
        print(a, end=' ')
        a, b = b, a + b
    print()
