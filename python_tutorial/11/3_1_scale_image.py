import argparse
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scale image tool')
    parser.add_argument('i', help='Input path')
    parser.add_argument('o', help='Output path')
    parser.add_argument('s', help='Scale')
    args = parser.parse_args()
    scale = float(args.s)
    try:
        image = Image.open(args.i)
        resized_image = image.resize((int(image.width * scale), int(image.height * scale)))
        resized_image.save(args.o)
    except Exception as e:
        print(e)
