import numpy
import os
import wave
from matplotlib import pyplot

# https://github.com/Jakobovski/free-spoken-digit-dataset
# Audio: pcm_s16le ([1][0][0][0] / 0x0001), 8000 Hz, mono, s16, 128 kb/s

def parse():
    base_url = '/Users/admin/Downloads/free-spoken-digit-dataset/recordings'
    audio_data_list = []
    audio_label_list = []
    for root, _, files in os.walk(base_url):
        for file in files:
            label = file.split('_')[0]
            audio_label_list.append(label)
            file_path = os.path.join(root, file)
            # print(file_path)
            with wave.open(file_path, 'rb') as wave_file:
                frame_count = wave_file.getnframes()
                frames = wave_file.readframes(frame_count)
                audio_data = numpy.frombuffer(frames, dtype=numpy.int16)
                audio_data_list.append(audio_data)
    # 填充音频数据，使所有数据长度一致
    max_length = max(len(data) for data in audio_data_list)
    audio_data_array = numpy.array([
        numpy.pad(data, (0, max_length - len(data)), 'constant', constant_values=0)
        for data in audio_data_list
    ])
    audio_label_array = numpy.array(audio_label_list, dtype=object)
    print(audio_data_array.shape, audio_label_array.shape)
    return audio_data_array, audio_label_array

if __name__ == '__main__':
    audio_data_array, audio_label_array = parse()
    pyplot.plot(range(0, len(audio_data_array[0])), audio_data_array[0])
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.06)
    pyplot.show()
