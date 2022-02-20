import taichi as ti
from numpy import linspace, array
from scipy.interpolate import pchip_interpolate
import tqdm

ti.init(arch=ti.cpu)

# canvas size
width = 1024
height = 720

# initial config
center_x = -0.77568377
center_y = 0.13646737
zoom = 150

# misc
zoom_rate = 1.2
max_iter = 1000
colormap_size = 1000
colormap = pchip_interpolate(
    [0, 0.16, 0.42, 0.6425, 0.8575, 1],
    array([[0, 7, 100], [32, 107, 203], [237, 255, 255], [255, 170, 0], [0, 2, 0], [0, 7, 100]]) / 255,
    linspace(0, 1, colormap_size)
    ).flatten()

pixels = ti.Vector.field(3, dtype=float, shape=(width, height))
# gui = ti.GUI("Mandelbrot Viewer", res=(width, height))

@ti.func
def iteration(x, y):
    c = ti.Vector([x, y])
    z = c
    count = 1.0
    while z.norm() <= 2 and count < max_iter:
        # z = z^2 + c
        z = ti.Vector([z[0]**2 - z[1]**2, z[0] * z[1] * 2]) + c
        count += 1.0
    if count < max_iter:
        # smooth color
        count += 1.0 - ti.log(ti.log(ti.cast(z.norm(), ti.f32)) / ti.log(2)) / ti.log(2)
    return count

@ti.kernel
def paint(center_x: ti.f64, center_y: ti.f64, zoom: ti.f64, colormap: ti.ext_arr()):
    for i, j in pixels:
        x = center_x + (i - width / 2 + 0.5) / zoom
        y = center_y + (j - height / 2 + 0.5) / zoom
        index = int(iteration(x, y) / max_iter * colormap_size)
        for k in ti.static(range(3)):
            pixels[i, j][k] = colormap[3 * index + k]

# GUI
# gui.fps_limit = 10
cnt = 0
result_dir = './mandelbrot_results'
video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)

for i in tqdm.tqdm(range(500)):
    zoom_new = zoom*1.05
    zoom = zoom_new
    paint(center_x, center_y, zoom, colormap)
    pixels_img = pixels.to_numpy()
    video_manager.write_frame(pixels_img)

print('Exporting videos...')
video_manager.make_video(mp4=True)
print('Done.')