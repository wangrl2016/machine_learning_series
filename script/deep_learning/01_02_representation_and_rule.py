from config import *
import matplotlib.pyplot as pyplot
import draw

if __name__ == '__main__':
    figure, axis = draw.init()
    draw.title(axis, '1.2 表示与规则')
    draw.subtitle(axis, '1. 文本如何表示')
    draw.text(axis, 'Unicode 标准是由 Unicode 联盟维护的文本编码标准，旨在支持使用世界上所有可数字化的书写系统的文本。', index=0)
    for i in range(6, 18):
        ax = pyplot.subplot(3, 6, i + 1)
        pyplot.title('Hello', fontsize=MEDIUM_SIZE)
        pyplot.axis('off')
    draw.save('02_01_unicode_standard.png')

    figure, axis = draw.init()
    draw.title(axis, '1.2 表示与规则')
    draw.subtitle(axis, '2. 图片如何表示')
    draw.image_h_center(axis, 'res/deep_learning/png_channel.png', bottom=LARGE_OFFSET, scale=1.5)
    draw.save('02_02_png_channel_1.png')

    figure, axis = draw.init()
    draw.title(axis, '1.2 表示与规则')
    draw.subtitle(axis, '2. 图片如何表示')
    draw.image_h_center(axis, 'res/deep_learning/png_palette.png', bottom=LARGE_OFFSET, scale=1.2)
    draw.save('02_02_png_palette_2.png')

    figure, axis = draw.init()
    draw.title(axis, '1.2 表示与规则')
    draw.subtitle(axis, '3. 视频如何表示')
    draw.text(axis, '视频的帧率：30', index=0)
    draw.save('02_02_video_3.png')

    figure, axis = draw.init()
    draw.title(axis, '1.2 表示与规则')
    draw.subtitle(axis, '4. 声音如何表示')
    draw.text(axis, '声音要素：声道、频率、采样率、幅度', index=0)
    draw.image_h_center(axis, 'res/deep_learning/wave_draw.png', bottom=LARGE_OFFSET)
    draw.save('02_02_wave_draw_4.png')
