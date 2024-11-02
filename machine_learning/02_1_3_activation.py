import keras
import numpy

def get_img_array(img_path, target_size):
    img = keras.utils.load_img(
        img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    return numpy.expand_dims(array, axis=0)

if __name__ == '__main__':
    # Build a ResNet50V2 model loaded with pre-trained ImageNet weights
    model = keras.applications.ResNet50V2(weights='imagenet')
    model.summary()

    img_path = keras.utils.get_file(
        fname='cat.jpg',
        origin='https://img-datasets.s3.amazonaws.com/cat.jpg')
    img_tensor = get_img_array(img_path, target_size=(224, 224))
    