
import file_util

pet_images_url = 'https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz'
pet_annotations_url = 'https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz'


file_util.get_file(pet_images_url, 'pet_images')
