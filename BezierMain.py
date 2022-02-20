import taichi as ti
from BezierBase import BezierBase
import tqdm

@ti.kernel
def getSliderValue(x : ti.f32) -> ti.i32:
    return ti.cast(x, ti.i32)

if __name__ == "__main__":
    ti.init()
    # 迭代步长
    t = 0

    # ui控制参数
    done = False
    start = False

    # 绘制窗口
    width = 1024
    height = 720
    gui = ti.GUI("Bezier Curve", (width, height))
    gui.button("Bezier Curve", event_name="startButton")
    degreeSetting = gui.slider("Degree", 1, 10, step=1)
    degreeSetting.value = 3
    start = True
    done = False
    t = 0
    # 初始化
    degree = 3
    degreeSetting.value = degree
    bezierBase = BezierBase(degree)
    bezierBase.setRandomBasePointPos()
    # GUI
    for i in tqdm.tqdm(range(6000)):
        gui.clear(0x83AF9B)
        if done:
            start = True
            done = False
            t = 0
            # 初始化
            degree+=1
            degreeSetting.value = degree
            bezierBase = BezierBase(degree)
            bezierBase.setRandomBasePointPos()

        # 计算并绘制贝塞尔曲线
        if start:
            # 绘制基点
            bezierBase.displayBasePoint(gui)
            if not done:
                if t < bezierBase.t_num:
                    bezierBase.computeBezier(t)
                    t += 1
                else:
                    done = True
            bezierBase.displayMidPoint(gui)
        filename = f'./bez_results/frames/{i:05d}.png'
        gui.show(filename)
