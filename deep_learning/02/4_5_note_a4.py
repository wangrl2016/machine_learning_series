import numpy
import wave

# A4 音符的频率
FREQUENCY = 440
# 采样率
SAMPLE_RATE = 48000
# 音频时长
DURATION = 2.0
# 声道数
CHANNEL_COUNT = 2

if __name__ == '__main__':
    t = numpy.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    waveform = 0.5 * numpy.sin(2 * numpy.pi * FREQUENCY * t)
    waveform = (waveform * numpy.iinfo(numpy.int16).max).astype(numpy.int16)
    data = numpy.empty(waveform.size * CHANNEL_COUNT, dtype=numpy.int16)
    data[0::2] = waveform   # 偶数位
    data[1::2] = waveform   # 奇数位
    with wave.open('temp/note_a4.wav', 'w') as wf:
        wf.setnchannels(CHANNEL_COUNT)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())
