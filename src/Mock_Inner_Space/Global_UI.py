import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


def init_window(width, height):
    # Initialize a window for display
    # 初始化显示窗口
    pygame.init()
    display = (width, height)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Set up the perspective projection
    # 设置透视投影
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    # Move the camera backwards so we can see the cube
    # 向后移动摄像头以便可以看到立方体
    glTranslatef(0.0, 0.0, -5)


def draw_cube():
    # Start drawing the cube
    # 开始绘制立方体
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x += 1
            # Set color for each vertex
            # 为每个顶点设置颜色
            glColor3fv(colors[x])
            # Draw each vertex
            # 绘制每个顶点
            glVertex3fv(vertices[vertex])
    glEnd()


def setup_scene():
    # Define vertices, surfaces, and colors for a cube
    # 为立方体定义顶点、表面和颜色
    global vertices, surfaces, colors
    vertices = (
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, -1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, -1, 1),
        (-1, 1, 1)
    )
    surfaces = (
        (0, 1, 2, 3),
        (3, 2, 7, 6),
        (6, 7, 5, 4),
        (4, 5, 1, 0),
        (1, 5, 7, 2),
        (4, 0, 3, 6)
    )
    colors = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 0, 0)
    )


def main():
    # Initialize the window and setup the scene
    # 初始化窗口并设置场景
    init_window(800, 600)
    setup_scene()
    clock = pygame.time.Clock()

    # Main event loop
    # 主事件循环
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Handle keyboard input to rotate the cube
        # 处理键盘输入以旋转立方体
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            glRotatef(1, 0, 1, 0)
        if keys[pygame.K_RIGHT]:
            glRotatef(-1, 0, 1, 0)
        if keys[pygame.K_UP]:
            glRotatef(1, 1, 0, 0)
        if keys[pygame.K_DOWN]:
            glRotatef(-1, 1, 0, 0)

        # Clear the screen and draw the cube
        # 清除屏幕并绘制立方体
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_cube()
        pygame.display.flip()
        clock.tick(60)  # Limit frames per second to 60


if __name__ == "__main__":
    main()
