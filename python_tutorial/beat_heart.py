import math
import random
from tkinter import Canvas, Tk

CANVAS_WIDTH = 960         # 画布宽度
CANVAS_HEIGHT = 540        # 画布高度
IMAGE_SCALE = 12.0          # 放大比例
HEART_COLOR = '#FF0000'     # 爱心颜色

def heart_function(t, shrink_ratio: float = IMAGE_SCALE):
    x = CANVAS_WIDTH / 2 + 16 * (math.sin(t) ** 3) * shrink_ratio
    y = CANVAS_HEIGHT / 2 + -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)) * shrink_ratio
    return int(x), int(y)

def scatter_inside(x, y, beta=0.15):
    ratio_x = - beta * math.log(random.random())
    ratio_y = - beta * math.log(random.random())
    dx = ratio_x * (x - CANVAS_WIDTH / 2)
    dy = ratio_y * (y - CANVAS_HEIGHT / 2)

    return x - dx, y - dy

def shrink(x, y, ratio):
    force = -1 / (((x - CANVAS_WIDTH / 2) ** 2 + (y - CANVAS_HEIGHT / 2) ** 2) ** 0.6)
    dx = ratio * force * (x - CANVAS_WIDTH / 2)
    dy = ratio * force * (y - CANVAS_HEIGHT / 2)
    return x - dx, y - dy

def beat_period_curve(p):
    return 12 * (2 * math.sin(4 * p)) / (2 * math.pi)

class Heart:
    def __init__(self, generate_frame=20):
        self._points = set()                        # 原始爱心坐标集合
        self._edge_diffusion_points = set()         # 边缘扩散效果点坐标集合
        self._center_diffusion_points = set()       # 中心扩散效果点坐标集合
        self.all_points = {}                        # 每帧动态点坐标
        self.build(2000)

        self.random_halo = 1000

        self.generate_frame = generate_frame
        for frame in range(generate_frame):
            self.calc(frame)

    def build(self, number):
        # 爱心
        for _ in range(number):
            t = random.uniform(0, 2 * math.pi)
            x, y = heart_function(t)
            self._points.add((x, y))

        # 爱心内扩散
        for _x, _y in list(self._points):
            for _ in range(3):
                x, y = scatter_inside(_x, _y, 0.05)
                self._edge_diffusion_points.add((x, y))

        # 爱心内再次扩散
        point_list = list(self._points)
        for _ in range(6000):
            x, y = random.choice(point_list)
            x, y = scatter_inside(x, y, 0.17)
            self._center_diffusion_points.add((x, y))

    @staticmethod
    def calc_position(x, y, ratio):
        force = 1 / (((x - CANVAS_WIDTH / 2) ** 2 + (y - CANVAS_HEIGHT / 2) ** 2) ** 0.520)  # 魔法参数
        dx = ratio * force * (x - CANVAS_WIDTH / 2) + random.randint(-1, 1)
        dy = ratio * force * (y - CANVAS_HEIGHT / 2) + random.randint(-1, 1)
        return x - dx, y - dy

    def calc(self, generate_frame):
        ratio = 10 * beat_period_curve(generate_frame / 10 * math.pi)  # 圆滑的周期的缩放比例
        halo_radius = int(4 + 6 * (1 + beat_period_curve(generate_frame / 10 * math.pi)))
        halo_number = int(3000 + 4000 * abs(beat_period_curve(generate_frame / 10 * math.pi) ** 2))
        all_points = []

        heart_halo_point = set()
        for _ in range(halo_number):
            t = random.uniform(0, 4 * math.pi)
            x, y = heart_function(t, shrink_ratio=11.5)
            x, y = shrink(x, y, halo_radius)
            if (x, y) not in heart_halo_point:
                heart_halo_point.add((x, y))
                x += random.randint(-14, 14)
                y += random.randint(-14, 14)
                size = random.choice((1, 2, 2))
                all_points.append((x, y, size))

        for x, y in self._points:
            x, y = self.calc_position(x, y, ratio)
            size = random.randint(1, 3)
            all_points.append((x, y, size))

        for x, y in self._edge_diffusion_points:
            x, y = self.calc_position(x, y, ratio)
            size = random.randint(1, 2)
            all_points.append((x, y, size))

        for x, y in self._center_diffusion_points:
            x, y = self.calc_position(x, y, ratio)
            size = random.randint(1, 2)
            all_points.append((x, y, size))

        self.all_points[generate_frame] = all_points

    def render(self, render_canvas, render_frame):
        for x, y, size in self.all_points[render_frame % self.generate_frame]:
            render_canvas.create_rectangle(x, y, x + size, y + size, width=0, fill=HEART_COLOR)

def draw(main: Tk, render_canvas: Canvas, render_heart: Heart, render_frame=0):
    render_canvas.delete('all')
    render_heart.render(render_canvas, render_frame)
    main.after(160, draw, main, render_canvas, render_heart, render_frame + 1)

if __name__ == '__main__':
    root = Tk()
    root.title('Beat Heart')
    canvas = Canvas(root, bg='black', width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
    canvas.pack()
    heart = Heart()
    draw(root, canvas, heart)
    root.mainloop()
