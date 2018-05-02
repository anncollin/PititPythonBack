# -*- coding: utf-8 -*-


from pygame import Surface, image, draw, font, freetype, Color
from board import Board
from collections import defaultdict
from autoplay import *
from MDP import *
import numpy as np
import os
import time


def draw_board(board, dice, expec, ref_expec, directory, filename, surf=None, render=False):
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
    orange_red = (255, 69, 0)
    pale_turquoise = (175, 238, 238)


    die_color = [black, forest_green, orange, red]
    tile_color = [lime_green, sky_blue, pink, peru]
    tile_emoji = [u"👟", u"⏮", u"◀", u"🔂", u"🏁"]  # 🌀 🔙 🕸

    surfX = 2760
    surfY = 480
    surfM = 20
    if surf is None:
        surf = Surface((surfX, surfY))
    else:
        surfX //= 2
        surfY //= 2
        surfM //= 2
    # white background
    surf.fill(white)

    # squares coordinates
    size = (surfX - 2*surfM) / 16
    x_co = surfM + np.append(1.5*size*np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                             2.0 * size * np.array([3, 4, 5, 6]))
    y_co = surfM + 1.5*size*np.array([0]*11 + [1]*4)
    co_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 10, 11, 12, 13]
    nmb_offset_x = 9
    nmb_offset_y = 7
    tile_emoji_offset = np.split((np.array([8+nmb_offset_x, 11+nmb_offset_y,
                                            12+nmb_offset_x, 8+nmb_offset_y,
                                            10.5+nmb_offset_x, 6+nmb_offset_y,
                                            8+nmb_offset_x, 3+nmb_offset_y,
                                            10, 10]))/100*size, 5)

    # draw cases
    def print_case(surf, x, y, values):
        def print_text(text, pos, color, emoji=False):
            text = str(text)
            ft = freetype.Font("seguisym.ttf", round(size*0.3)) if emoji else freetype.SysFont('Calibri', round(size*0.2))
            text = ft.render(text, fgcolor=color)
            surf.blit(text[0], pos)

        tile = values[2]
        emoji_offset = tile_emoji_offset[tile]
        draw.rect(surf, grey, (x, y, size, size), 0)

        # draw.rect(surf, tile_color[tile], (x, y, size*0.5, size*0.4), 0)
        draw.rect(surf, black, (x, y, size * 0.5, size * 0.4), 0)
        draw.rect(surf, black, (x, y, size*0.5, size*0.4), 2)
        print_text(values[0], (x+0.045*size, y+0.05*size), white)
        print_text(tile_emoji[tile], (x + +emoji_offset[0], y + +emoji_offset[1]), tile_color[tile], emoji=True)

        # Die
        def draw_die_face(face, x, y):
            face_width = size*0.4
            radius = face_width/8.0
            center = (int(round(x+face_width/2.0)), int(round(y+face_width/2.0)))
            radius = int(round(radius))
            face_width = round(int(face_width))

            if face != 2:
                draw.circle(surf, white, center, radius)
                draw.circle(surf, black, center, radius, 2)
            if face != 1:
                draw.circle(surf, white, (center[0]-radius*2, center[1]+radius*2), radius)
                draw.circle(surf, black, (center[0] - radius * 2, center[1] + radius * 2), radius, 2)

                draw.circle(surf, white, (center[0]+radius*2, center[1]-radius*2), radius)
                draw.circle(surf, black, (center[0] + radius * 2, center[1] - radius * 2), radius, 2)

        die = values[1]
        nx = x+0.1*size
        ny = y+size*0.5
        draw.rect(surf, die_color[die], (nx, ny, size*0.4, size*0.4), 0)
        draw.rect(surf, black, (nx, ny, size * 0.4, size * 0.4), 3)
        # print_text(die, (nx+0.1*size, ny+0.05*size), white)
        if die > 0: draw_die_face(die, nx, ny)

        # Gradient bar
        def draw_gradient(c1, c2, x, y, width, height, proportion):
            act_h = 0
            step = (c2-c1)/height
            draw.rect(surf, (black if proportion <= 1 else red), (x-1, y-height, width+3, height+3), 0)
            while (act_h/height - proportion) < 0.01 and act_h <= height:
                act_c = c1 + step*act_h
                act_y = y - act_h
                draw.line(surf, act_c, (x, act_y), (x + width, act_y), 1)
                act_h += 1

        proportion = values[3]
        nx = x + size*0.6
        ny = y + size*0.9
        draw_gradient(np.array(pale_turquoise), np.array(orange_red), nx, ny, size*0.3, size*0.7, proportion)

        # Outer border
        draw.rect(surf, black, (x, y, size, size), 3)

    freetype.init()
    for x, y, ind in zip(x_co, y_co, co_map):
        if ind != 14:
            print_case(surf, x, y, [ind+1, dice[ind], board[ind], expec[ind]/ref_expec])
        else:
            draw.rect(surf, black, (x, y, size, size), 0)
            draw.rect(surf, black, (x, y, size, size), 3)
            ft = freetype.Font("seguisym.ttf", 140)
            text = ft.render(tile_emoji[4], fgcolor=white)
            surf.blit(text[0], (x+tile_emoji_offset[4][0], y+tile_emoji_offset[4][1]))

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
    if not render:
        if not os.path.exists(directory):
            os.makedirs(directory)
        image.save(surf, os.path.join(directory, filename))
        return None
    else:
        return image


if __name__ == "__main__":
    seed(456)
    boards = [
        Board(defaultdict(list), circling=True), Board(defaultdict(list), circling=False),
              Board({"trap1": [10], "trap2": [4, 5, 8], "trap3":[]}, circling=False),
              Board({"trap1": [9, 13], "trap2": [8, 12, 4], "trap3": [1, 10, 11]}, circling=True)
              ]
    names = ["p1", "p2", "p3", "p4"]
    file = open("Simulation_results.txt","w")
    for board, name in zip(boards, names):
        file.write("New Plateau : %s \n" % name)

        # Markov agent
        start = time.clock()
        mdp_expec, mdp_dice = markov_decision(board)
        elapsed = (time.clock() - start) * 1000
        mdp_expec = play_strategy(board, lambda x: mdp_dice[x]-1, n_games=100000)
        draw_board([x.type for x in board.board], mdp_dice, mdp_expec, mdp_expec[0], name, "mdp" + ".png")
        file.write("Markov Agent \n")
        file.write("Computational time %s ms \n" % "{0:.2f}".format(elapsed)) 
        file.write(' & '.join(["{0:.2f}".format(x) for x in mdp_expec]) + ' \\\\' + '\n')
        
        # Homemade Agent 
        start = time.clock()
        ag_dice = find_strategy(board, n_games=1000, very_intelligent=True)
        elapsed = (time.clock() - start)*1000 
        ag_expec = play_strategy(board, lambda x: ag_dice[x], n_games=100000)
        draw_board([x.type for x in board.board], [d+1 for d in ag_dice], ag_expec, ag_expec[0], name, "ag" + ".png")
        file.write("Homemade Agent \n")
        file.write("Computational time %s ms \n" % "{0:.2f}".format(elapsed)) 
        file.write(' & '.join(["{0:.2f}".format(x) for x in ag_expec]) + ' \\\\' + '\n')

        # Suboptimal Agent 
        start = time.clock()
        sub_dice = find_strategy(board, n_games=1000, very_intelligent=False)
        elapsed = (time.clock() - start)*1000
        sub_expec = play_strategy(board, lambda x: sub_dice[x], n_games=100000)
        draw_board([x.type for x in board.board], [d+1 for d in sub_dice], sub_expec, sub_expec[0], name, "sub" + ".png")
        file.write("Suboptimal Agent \n")
        file.write("Computational time %s ms \n" % "{0:.2f}".format(elapsed)) 
        file.write(' & '.join(["{0:.2f}".format(x) for x in sub_expec]) + ' \\\\' + '\n')

        # Random Agent 
        rand_expec = play_strategy(board, lambda x: randrange(3), n_games=10000)
        draw_board([x.type for x in board.board], [0]*15, rand_expec, ag_expec[0], name, "rand" + ".png")
        file.write("Random Agent \n ")
        file.write(' & '.join(["{0:.2f}".format(x) for x in rand_expec]) + ' \\\\' + '\n \n')
        
    file.close()

