import tensorflow as tf
import einops
import pathlib
import keras
import numpy

class ShapeChecker():
    def __init__(self):
        # Keep a cache of every axis-name seen.
        self.shapes = {}
    
    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return
        
        parsed = einops.parse_shape(tensor, names)
        
        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)

            if (broadcast and new_dim == 1):
                continue
            
            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue
            
            if new_dim != old_dim:
                raise ValueError(f'Shape mismatch for dimension: {name}\n'
                                 f'    found: {new_dim}\n'
                                 f'    expected: {old_dim}')

def load_data(path):
    text = path.read_text(encoding='utf-8')
    
    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]
    
    context = numpy.array([context for target, context in pairs])
    target = numpy.array([target for target, context in pairs])
    
    return target, context

if __name__ == '__main__':
    path_to_zip = keras.utils.get_file(
        'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True)
    
    # Dir 'spa-eng_extracted/' for some computer. 
    path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng_extracted/spa-eng/spa.txt'
    
    target_raw, context_raw = load_data(path_to_file)
    print(context_raw[-1])
    print(target_raw[-1])
