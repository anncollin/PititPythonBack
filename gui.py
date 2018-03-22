from pygame import Surface, image, draw, font
from board import Board
from math import sqrt
import numpy as np

board = [x.type for x in Board({'trap1': [1, 10], 'trap2': [3, 5, 7]}).board]
dice = [2, 1, 3, 3, 2, 3, 3, 3, 2, 1, 1, 3, 2, 1, 1]
expec = [11.698915143254899, 11.032249954001127, 9.032252820017574, 9.701159919483432, 8.43010450515775, 7.401431295786926, 5.643368090782532, 4.232526126289438, 2.5, 2.0, 4.833333333333332, 2.833333333333333, 2.5, 2.0, 0]
ref_expec = expec[0]

# Color initialization
red = (255, 0, 0)
green = (0, 255, 0)
orange = (255, 165, 0)
blue = (0, 0, 255)
dark_blue = (0, 0, 128)
sky_blue = (0, 191, 255)
white = (255, 255, 255)
black = (0, 0, 0)
pink = (255, 200, 200)
forest_green = (34, 139, 34)
peru = (205, 133, 63)
lime_green = (50, 205, 50)
grey = (205, 201, 201)

die_color = [black, forest_green, orange, red]
tile_color = [lime_green, sky_blue, pink, peru]


surfX = 2760
surfY = 440
surfM = 20
surf = Surface((surfX, surfY))

# white background
surf.fill(white)

# squares coordinates
size = (surfX - 2*surfM) / 16
x_co = surfM + np.append(1.5*size*np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                         2.0 * size * np.array([3, 4, 5, 6]))
y_co = surfM + 1.5*size*np.array([0]*11 + [1]*4)
co_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 10, 11, 12, 13]


# draw cases
def print_case(surf, x, y, values):
    def print_text(text, pos, color):
        text = str(text)
        ft = font.SysFont('Calibri', round(size/2))
        text = ft.render(text, True, color)
        surf.blit(text, pos)

    tile = values[2]
    draw.rect(surf, grey, (x, y, size, size), 0)

    draw.rect(surf, tile_color[tile], (x, y, size*0.5, size*0.4), 0)
    draw.rect(surf, black, (x, y, size*0.5, size*0.4), 2)
    print_text(values[0], (x+4, y+3), black)

    # Die
    die = values[1]
    nx = x+7
    ny = y+7+size*0.4
    draw.rect(surf, die_color[die], (nx, ny, size*0.4, size*0.4), 0)
    print_text(die, (nx+7, ny+3), white)

    # Gradient bar
    def draw_gradient(c1, c2, x, y, width, height):
        act_h = 0
        step = (c2-c1)/height
        draw.rect(surf, black, (x, y-height, width+1, height+1), 0)
        while act_h/height <= proportion:
            act_c = c1 + step*act_h
            act_y = y - act_h
            draw.line(surf, act_c, (x, act_y), (x + width, act_y), 1)
            act_h += 1

    proportion = values[3]
    nx = x + size*0.6
    ny = y + size*0.8
    draw_gradient(np.array(green), np.array(red), nx, ny, size*0.3, size*0.6)

    # Outer border
    draw.rect(surf, black, (x, y, size, size), 3)


font.init()
for x, y, ind in zip(x_co, y_co, co_map):
    if ind != 14:
        print_case(surf, x, y, [ind+1, dice[ind], board[ind], expec[ind]/ref_expec])
    else:
        draw.rect(surf, black, (x, y, size, size), 0)
        draw.rect(surf, black, (x, y, size, size), 3)


# draw lines
x_l = surfM + size + np.append(np.append(1.5 * size * np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                         2 * size * np.array([2, 3, 4, 5, 6])),
                               size * np.array([3, 14]))

y_l = surfM + size / 2 + np.append(np.append(1.5 * size * np.array([0]*10),
                                             1.5 * size * np.array([1]*5)),
                                   np.array([0]*2))
dir_l = [[1, 0]] * 10 + [[2, 0]] * 5 + [[2, 3]] + [[-2, 3]]
for x, y, d in zip(x_l, y_l, dir_l):
    draw.line(surf, black, (x, y), (x + d[0] * size/2, y + d[1] * size * 0.5), 3)


# save picture
image.save(surf, "test.png")
