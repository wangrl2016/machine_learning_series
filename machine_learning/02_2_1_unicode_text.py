
if __name__ == '__main__':
    # 输出 65，对应代码点 U+0041
    print(ord('A'))
    # 输出 '😀'
    print(chr(0x1F600))
    # 字符转换为代码点
    char = '😀'
    code_point = ord(char)
    print(f"Character: {char}, Code point: U+{code_point:04X}")
    # UTF-8 和 UTF-16 编码
    char = '😀'
    utf8_encoded = char.encode('utf-8').hex()
    print('UTF-8:', utf8_encoded)
    utf16_encoded = char.encode('utf-16').hex()
    print('UTF-16:', utf16_encoded)
    utf32_encoded = char.encode('utf-32').hex()
    print('UTF-32:', utf32_encoded)
