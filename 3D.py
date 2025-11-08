import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.image import imread
from skimage.transform import resize

W, H = 1000, 1000
img_canvas = np.ones((H, W, 3), dtype=np.float32)*0.6
A = (300, 650)
B = (500, 300)
C = (700, 650)
O = (500, 500)
r, r_small, R = 85, 70, 350

# отрисовка пикселя
def DrawPixel(px, py, color=(0, 0, 0), size=2):
    for dx in range(-size, size + 1):
        for dy in range(-size, size + 1):
            xx, yy = px + dx, py + dy
            if 0 <= xx < W and 0 <= yy < H:
                img_canvas[H - 1 - yy, xx] = color

# алгоритм Брезенхема для линии
def LineBresenham(x1, y1, x2, y2, color=(0, 0, 0), size=2, hidden_mask=None):
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
    err = dx - dy
    while True:
        if 0 <= x1 < W and 0 <= y1 < H:
            if hidden_mask is None or not hidden_mask[H - 1 - y1, x1]:
                DrawPixel(x1, y1, color, size)
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

# алгоритм Брезенхема для окружности
def CircleBresenham(xc, yc, r, color=(0, 0, 0), size=2):
    x, y = 0, r
    d = 3 - 2 * r
    while y >= x:
        for dx, dy in [(x, y), (y, x), (-x, y), (-y, x),
                       (x, -y), (y, -x), (-x, -y), (-y, -x)]:
            DrawPixel(xc + dx, yc + dy, color, size)
        x += 1
        if d > 0:
            y -= 1
            d += 4 * (x - y) + 10
        else:
            d += 4 * x + 6

#поворот точки
def RotatePoint(px, py, cx, cy, angle_deg):
    rad = math.radians(angle_deg)
    dx, dy = px - cx, py - cy
    x_new = cx + dx * math.cos(rad) - dy * math.sin(rad)
    y_new = cy + dx * math.sin(rad) + dy * math.cos(rad)
    return int(round(x_new)), int(round(y_new))

# провкрка принадлежности точки треугольнику
def InsideTriangle(px, py, A, B, C):
    (x1, y1), (x2, y2), (x3, y3) = A, B, C
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    a = ((y2 - y3)*(px - x3) + (x3 - x2)*(py - y3)) / denom
    b = ((y3 - y1)*(px - x3) + (x1 - x3)*(py - y3)) / denom
    c = 1 - a - b
    return (0 <= a <= 1) and (0 <= b <= 1) and (0 <= c <= 1)

def ScaleTriangle(A, B, C, factor=1.12):
    cx = (A[0] + B[0] + C[0]) / 3
    cy = (A[1] + B[1] + C[1]) / 3
    def scale_point(p):
        return (
            int(cx + (p[0] - cx) * factor),
            int(cy + (p[1] - cy) * factor)
        )
    return scale_point(A), scale_point(B), scale_point(C)

A, B, C = [RotatePoint(x, y, O[0], O[1], 180) for (x, y) in [A, B, C]]
A2, B2, C2 = ScaleTriangle(A, B, C, 1.12)

# алгоритм  отсечения отрезка(Кирус–Бек)
def CyrusBeck(p1, p2, polygon):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    t_in, t_out = 0, 1
    n = len(polygon)
    for i in range(n):
        Ax, Ay = polygon[i]
        Bx, By = polygon[(i + 1) % n]
        nx, ny = Ay - By, Bx - Ax
        wx, wy = x1 - Ax, y1 - Ay
        Dn = dx * nx + dy * ny
        Wn = wx * nx + wy * ny
        if Dn == 0:
            if Wn < 0:
                return None
            else:
                continue
        t = -Wn / Dn
        if Dn > 0:
            t_in = max(t_in, t)
        else:
            t_out = min(t_out, t)
        if t_in > t_out:
            return None
    if t_in > 1 or t_out < 0:
        return None
    return t_in, t_out


img_path = r"C:\python3D\image.jpg"
try:
    image = imread(img_path)
    if image.dtype != np.uint8 and image.max() > 1.0:
        image = image / 255.0
    resized = resize(image, (H, W), anti_aliasing=True)

    mask = np.zeros((H, W), dtype=bool)
    for x in range(W):
        for y in range(H):
            if InsideTriangle(x, y, A, B, C):
                dist = math.sqrt((x - O[0])**2 + (y - O[1])**2)
                if dist > r:
                    mask[H - 1 - y, x] = True
    img_canvas[mask] = resized[mask]
except FileNotFoundError:
    print("no image found")

LineBresenham(*A, *B, size=3)
LineBresenham(*B, *C, size=3)
LineBresenham(*C, *A, size=3)

# пунктирная линия
def Dashed(p1, p2, dash=60, gap=20):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    total = math.hypot(dx, dy)
    pos = 0
    while pos < total:
        end = min(pos + dash, total)
        t1, t2 = pos / total, end / total
        xs, ys = int(x1 + dx * t1), int(y1 + dy * t1)
        xe, ye = int(x1 + dx * t2), int(y1 + dy * t2)
        LineBresenham(xs, ys, xe, ye, size=2)
        pos += dash + gap

Dashed(A2, B2)
Dashed(B2, C2)
Dashed(C2, A2)

for angle in range(0, 360, 60):
    for a in np.linspace(angle, angle + 30, 300):
        x = int(O[0] + r_small * math.cos(math.radians(a)))
        y = int(O[1] + r_small * math.sin(math.radians(a)))
        DrawPixel(x, y, size=2)

CircleBresenham(O[0], O[1], r, size=2)

# большая дуга
for a in np.linspace(-50, 180, 600):
    x = int(O[0] + R * math.cos(math.radians(a + 180)))
    y = int(O[1] + R * math.sin(math.radians(a + 180)))
    DrawPixel(x, y, size=2)

seg_start = (80, 700)
seg_end = (950, 475)
clip = CyrusBeck(seg_start, seg_end, [A, B, C])
if clip:
    t_in, t_out = clip
    dx, dy = seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]
    p_in = (int(seg_start[0] + dx * t_in), int(seg_start[1] + dy * t_in))
    p_out = (int(seg_start[0] + dx * t_out), int(seg_start[1] + dy * t_out))
    LineBresenham(seg_start[0], seg_start[1], p_in[0], p_in[1], size=2)
    LineBresenham(p_out[0], p_out[1], seg_end[0], seg_end[1], size=2)
else:
    LineBresenham(*seg_start, *seg_end, size=2)

plt.figure(figsize=(10, 10))
plt.imshow(img_canvas)
plt.show()
