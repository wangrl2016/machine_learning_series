import manim
import numpy

manim.config.pixel_width = 1920
manim.config.pixel_height = 1080
manim.config.frame_rate = 30

# 创建场景类
class SingleElementIndexing(manim.Scene):
    def construct(self):
        self.camera.background_color = '#F5F5F5'
        # 创建 NumPy 数组
        arr = numpy.arange(10)
        array_elements = [manim.Text(str(i), color=manim.BLACK) for i in arr]
        for element in array_elements:
            element.scale(0.8)
        array_group = manim.VGroup(*[manim.Square(side_length=0.8,stroke_color=manim.BLACK)
                                     .add(elem) for elem in array_elements])
        array_group.arrange(manim.RIGHT, buff=0.5)
        
        self.play(manim.FadeIn(array_group))
        self.wait(5)
        
        element_text1 = manim.Text('arr[2] = ?', color=manim.BLACK)
        element_text1.shift(manim.UP * 2)
        self.play(manim.Write(element_text1))
        self.wait(3)
        self.remove(element_text1)
        self.wait(1)

        # 正下标
        pos_index_arr = numpy.arange(0, 10, 1)
        pos_index_elements = [manim.Text(str(i), color=manim.BLACK) for i in pos_index_arr]
        for element in pos_index_elements:
            element.scale(0.8)
        pos_index_group = manim.VGroup(*[manim.Square(side_length=0.8,stroke_color='#F5F5F5')
                                     .add(elem) for elem in pos_index_elements])
        pos_index_group.arrange(manim.RIGHT, buff=0.5)
        pos_index_group.shift(manim.DOWN)
        self.play(manim.FadeIn(pos_index_group))
        self.wait(5)
        
        highlight_box = manim.SurroundingRectangle(array_group[2], color=manim.RED, buff=0.2)
        self.play(manim.Create(highlight_box))
        self.wait(3)

        element_text2 = manim.Text('arr[2] = 2', color=manim.BLACK)
        element_text2.shift(manim.UP * 2)
        self.play(manim.Write(element_text2))
        self.wait(3)
        
        self.play(manim.FadeOut(element_text2), manim.FadeOut(highlight_box))
        self.wait(1)

        element_text3 = manim.Text('arr[-2] = ?', color=manim.BLACK)
        element_text3.shift(manim.UP * 2)
        self.play(manim.Write(element_text3))
        self.wait(3)
        self.remove(element_text3)
        self.wait(1)
        
        # 负下标
        neg_index_arr = numpy.arange(-10, 0, 1)
        neg_index_elements = [manim.Text(str(i), color=manim.BLACK) for i in neg_index_arr]
        for element in neg_index_elements:
            element.scale(0.8)
        neg_index_group = manim.VGroup(*[manim.Square(side_length=0.8,stroke_color='#F5F5F5')
                                     .add(elem) for elem in neg_index_elements])
        neg_index_group.arrange(manim.RIGHT, buff=0.5)
        neg_index_group.shift(manim.DOWN * 2)
        self.play(manim.FadeIn(neg_index_group))
        self.wait(5)
        
        highlight_box = manim.SurroundingRectangle(array_group[8], color=manim.RED, buff=0.2)
        self.play(manim.Create(highlight_box))
        self.wait(3)

        element_text4 = manim.Text('arr[-2] = 8', color=manim.BLACK)
        element_text4.shift(manim.UP * 2)
        self.play(manim.Write(element_text4))
        self.wait(3)

        self.play(manim.FadeOut(element_text4), manim.FadeOut(highlight_box))
        self.wait(1)
        
        self.play(manim.FadeOut(array_group, pos_index_group, neg_index_group))
        self.wait(1)
    
        # 二维矩阵
        data = [[str(i) for i in range(5)], [str(i) for i in range(5, 10)]]
        table = manim.Table(
            data, 
            include_outer_lines=True,
            line_config={'stroke_color': manim.BLACK},
            element_to_mobject=lambda elem: manim.Text(elem, color=manim.BLACK)
        )
        table.scale(0.8).move_to(manim.ORIGIN)
        self.play(manim.Create(table))
        self.wait(3)
        
        cell_index_list = [(1, 1), (2, 1), (2, 2), (2, 3), (2, 4)]
        for cell_index in cell_index_list:
            cell = table.get_cell(cell_index)
            highlight_box = manim.SurroundingRectangle(cell, color=manim.RED, buff=0.1)
            self.play(manim.Create(highlight_box))
            self.wait(1)
            self.remove(highlight_box)

