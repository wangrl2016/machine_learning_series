import keras
import numpy
import os
from matplotlib import pyplot

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
    
    pyplot.axis('off')
    pyplot.imshow(img_tensor[0].astype('uint8'))
    pyplot.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0)
    pyplot.show()
    
    # 对猫的图片进行预测
    predictions = model.predict(keras.applications.resnet_v2.preprocess_input(img_tensor.copy()))
    decoded_predictions = keras.applications.resnet_v2.decode_predictions(predictions, top=3)[0]
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f'{i+1}: {label} ({score:.2f})')
    
    # 绘制预测的中间层输出
    layer_outputs = []
    layer_names = []
    for index, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            layer_outputs.append(layer.output)
            layer_names.append(f'Layer {index+1}: {layer.name}')

    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    for layer_name, activation in zip(layer_names, activations):
        print(activation.shape)
        pyplot.matshow(activation[0, :, :, 0])
        pyplot.title(layer_name)
        pyplot.savefig(os.path.join('temp', layer_name + '.png'))
        # pyplot.show()