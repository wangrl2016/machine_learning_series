import matplotlib.pyplot  as pyplot
import numpy
import os
import requests
import struct
import wave

wave_file_url = 'https://freewavesamples.com/files/Alesis-Fusion-Nylon-String-Guitar-C4.wav'

if __name__ == '__main__':
    filename = wave_file_url.split('/')[-1]
    response = requests.get(wave_file_url)
    if response.status_code == 200:
        with open(os.path.join('temp', filename), 'wb') as fp:
            fp.write(response.content)
    else:
        print('Failed to download file, status code:', response.status_code)

    # 1. Use python wave module to decode wave file.
    with wave.open(os.path.join('temp', filename)) as wave_file:
        metadata = wave_file.getparams()
        print(metadata)
        samplerate = wave_file.getframerate()
        num_channel = wave_file.getnchannels()
        frames = wave_file.readframes(metadata.nframes)
        # only draw single channel
        frames = numpy.frombuffer(frames, dtype='int16')[::num_channel]
        time = numpy.linspace(0, len(frames) / samplerate, num = len(frames))

        # create a new figure
        pyplot.figure()
        pyplot.title('Sound Wave')
        pyplot.xlabel('Time')
        pyplot.plot(time, frames)
        pyplot.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.06)
        pyplot.savefig('temp/wave_draw_1.png', dpi=300)
        pyplot.show()
        pyplot.close('all')

    # 2. Hard-code wave decoder.
    with open(os.path.join('temp', filename), 'rb') as fp:
        # 1 - 12 bytes
        chunk_id, chunk_size, format = struct.unpack('<4sI4s', fp.read(12))
        if chunk_id != b'RIFF' or format != b'WAVE':
            raise ValueError('Not a valid WAVE file')
        
        # 12 - 20 bytes
        subchunk1_id, subchunk1_size = struct.unpack('<4sI', fp.read(8))
        if subchunk1_id != b'fmt ':
            raise ValueError('Invalid fmt chunk')
        
        # 20 - 36 bytes
        audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample = \
            struct.unpack('<HHIIHH', fp.read(subchunk1_size)[:16])
        print('Audio format', audio_format)
        print('Number of channels', num_channels)
        print('Sample rate', sample_rate)
        print('Byte rate', byte_rate)
        print('Block align', block_align)
        print('Bits per sample', bits_per_sample)

        # 36 - 44 bytes
        subchunk2_id, subchunk2_size = struct.unpack('<4sI', fp.read(8))
        if subchunk2_id != b'data':
            raise ValueError('Invalid subchunk2 id')
        
        data = fp.read(subchunk2_size)
        if bits_per_sample == 8:
            dtype = numpy.int8
        elif bits_per_sample == 16:
            dtype = numpy.int16
        else:
            raise ValueError('Unsupported bit depth', bits_per_sample)
        audio_data = numpy.frombuffer(data, dtype=dtype)[::num_channel]
        time = numpy.linspace(0, len(audio_data) / sample_rate, num = len(audio_data))
        pyplot.figure()
        pyplot.title('Sound Wave')
        pyplot.plot(time, audio_data)
        pyplot.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.06)
        pyplot.savefig('temp/wave_draw_2.png', dpi=300, transparent=True)
        pyplot.show()
        pyplot.close('all')
