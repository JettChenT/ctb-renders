import taichi as ti
import math

ti.init(arch=ti.gpu)

n = 720
RATIO = 2
tmp = ti.field(dtype=float, shape=(n * RATIO, n))
pixels = ti.field(dtype=ti.uint8, shape=(n * RATIO, n, 3))


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])


@ti.func
def get_rgb_val(n, k):
    k = k%3+1
    return n * (256 ** k) % 256


@ti.kernel
def paint(mx:float, my:float):
    for i, j in tmp:  # Parallized over all pixels
        c = ti.Vector([mx-0.5, my-0.5])*2
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        tmp[i, j] = 1 - iterations * 0.02
    for i, j, k in pixels:
        pixels[i, j, k] = get_rgb_val(tmp[i, j], k + 1)


gui = ti.GUI("Julia Set", res=(n * RATIO, n))

def clip(n):
    return (n+1)/2

for i in range(1000):
    mouse_x, mouse_y = clip(math.cos(i/75))/2+0.25, clip(math.sin(i/75))/2+0.25
    paint(mouse_x, mouse_y)
    if(i%60==0):
        print(mouse_x,mouse_y)
    gui.set_image(pixels)
    gui.circle(ti.Vector([0.5,0.5]),0xff0000,5)
    gui.circle(ti.Vector([mouse_x, mouse_y]), 0x00ff00, 5)
    gui.show()
