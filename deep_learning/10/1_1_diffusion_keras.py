import keras_hub
from matplotlib import pyplot
import numpy
import PIL

# https://keras.io/keras_hub/guides/stable_diffusion_3_in_keras_hub/
# 

def display_generated_images(images):
    """Helper function to display the images from the inputs.

    This function accepts the following input formats:
    - 3D numpy array.
    - 4D numpy array: concatenated horizontally.
    - List of 3D numpy arrays: concatenated horizontally.
    """
    display_image = None
    if isinstance(images, numpy.ndarray):
        if images.ndim == 3:
            display_image = PIL.Image.fromarray(images)
        elif images.ndim == 4:
            concated_images = numpy.concatenate(list(images), axis=1)
            display_image = PIL.Image.fromarray(concated_images)
    elif isinstance(images, list):
        concated_images = numpy.concatenate(images, axis=1)
        display_image = PIL.Image.fromarray(concated_images)

    if display_image is None:
        raise ValueError("Unsupported input format.")

    pyplot.figure(figsize=(10, 10))
    pyplot.axis("off")
    pyplot.imshow(display_image)
    pyplot.show()
    pyplot.close()

if __name__ == '__main__':
    backbone = keras_hub.models.StableDiffusion3Backbone.from_preset(
        "stable_diffusion_3_medium", image_shape=(512, 512, 3), dtype="float16")
    preprocessor = keras_hub.models.StableDiffusion3TextToImagePreprocessor.from_preset(
        "stable_diffusion_3_medium")
    text_to_image = keras_hub.models.StableDiffusion3TextToImage(backbone, preprocessor)

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 2k"
    generated_image = text_to_image.generate(prompt)
    display_generated_images(generated_image)
