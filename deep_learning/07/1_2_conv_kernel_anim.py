import manim
import numpy

manim.config.pixel_width = 1920
manim.config.pixel_height = 1080
manim.config.frame_rate = 30

class ConvolutionAnimation(manim.Scene):
    def construct(self):
        self.camera.background_color = '#F5F5F5'
        input_matrix = [[1, 2, 3, 0, 1],
                        [4, 5, 6, 1, 0],
                        [7, 8, 9, 2, 3],
                        [2, 0, 1, 4, 5],
                        [3, 4, 5, 6, 7]]
        kernel = [[1, 0, -1],
                  [1, 0, -1],
                  [1, 0, -1]]
        output_matrix = numpy.zeros((3, 3))

        input_table = self.create_matrix_table(input_matrix)
        input_table.scale(0.5)
        input_table.move_to(manim.LEFT * 4)

        kernel_table = self.create_matrix_table(kernel)
        kernel_table.scale(0.5)
        kernel_table.next_to(input_table, manim.RIGHT, buff=1)

        output_table  = self.create_matrix_table(output_matrix)
        output_table.scale(0.5)
        output_table.next_to(kernel_table, manim.RIGHT, buff=1)

        input_title = manim.Text("Input", color=manim.BLACK).next_to(input_table, manim.UP)
        input_title.scale(0.5)
        kernel_title = manim.Text("Kernel", color=manim.BLACK).next_to(kernel_table, manim.UP)
        kernel_title.scale(0.5)
        output_title = manim.Text("Output", color=manim.BLACK).next_to(output_table, manim.UP)
        output_title.scale(0.5)
        self.play(manim.Create(input_table), manim.Write(input_title))
        self.play(manim.Create(kernel_table), manim.Write(kernel_title))
        self.play(manim.Create(output_table), manim.Write(output_title))
        self.wait(3)

        kernel_rect = manim.SurroundingRectangle(kernel_table, color=manim.BLUE)
        self.play(manim.Create(kernel_rect))
        self.wait(3)

        (h, w) = output_matrix.shape
        for i in range(h):
            for j in range(w):
                rect = manim.SurroundingRectangle(manim.VGroup(
                    input_table.get_entries()[i * 5 + j],
                    input_table.get_entries()[i * 5 + j + 1],
                    input_table.get_entries()[i * 5 + j + 2],
                    input_table.get_entries()[(i + 1) * 5 + j],
                    input_table.get_entries()[(i + 1) * 5 + j + 1],
                    input_table.get_entries()[(i + 1) * 5 + j + 2],
                    input_table.get_entries()[(i + 2) * 5 + j],
                    input_table.get_entries()[(i + 2) * 5 + j + 1],
                    input_table.get_entries()[(i + 2) * 5 + j + 2]
                ), color=manim.GREEN)
                self.play(manim.Transform(kernel_rect, rect))
                
                result = sum(input_matrix[i+ki][j+kj] * kernel[ki][kj]
                             for ki in range(3) for kj in range(3))
                print(result)
        self.wait(3)

    def create_matrix_table(self, matrix):
        return manim.Table(
            [[str(value) for value in row] for row in matrix],
            include_outer_lines=True,
            line_config={'stroke_color': manim.BLACK},
            element_to_mobject=lambda elem: manim.Text(elem, color=manim.BLACK),
        )
