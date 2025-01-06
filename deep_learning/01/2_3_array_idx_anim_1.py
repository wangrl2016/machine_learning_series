import manim

class ArrayIndexing(manim.Scene):
    def construct(self):
        array_elements = [manim.Text(str(i)) for i in range(5)]
        array_group = manim.VGroup(*[manim.Square(side_length=1).add(elem) for elem in array_elements])
        array_group.arrange(manim.RIGHT, buff=0.5)
        
        indices = manim.VGroup(*[manim.Text(f'{i}').scale(0.7) for i in range(5)])
        for index, square in zip(indices, array_group):
            index.next_to(square, manim.DOWN)
        
        # 显示数组和索引
        self.play(manim.FadeIn(array_group), manim.FadeIn(indices))
        self.wait()
        
        # 高亮数组中的元素
        highlight_box = manim.SurroundingRectangle(array_group[2], color=manim.YELLOW, buff=0.1)
        self.play(manim.Create(highlight_box))
        self.wait()
