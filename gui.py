# -*- coding: utf-8 -*-


from pygame import Surface, image, draw, font, freetype, Color
from board import Board
from collections import defaultdict
from autoplay import *
from MDP import *
import numpy as np
import os


def draw_board(board, dice, expec, ref_expec, directory, filename):
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
    tile_emoji = [u"👟", u"⏮", u"◀", u"🔂", u"🏁"] # 🌀 🔙 🕸

    surfX = 2760
    surfY = 480
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
        def print_text(text, pos, color, emoji=False):
            text = str(text)
            ft = freetype.Font("seguisym.ttf", 64) if emoji else freetype.SysFont('Calibri', round(size*0.4))
            text = ft.render(text, fgcolor=color)
            surf.blit(text[0], pos)

        tile = values[2]
        draw.rect(surf, grey, (x, y, size, size), 0)

        # draw.rect(surf, tile_color[tile], (x, y, size*0.5, size*0.4), 0)
        draw.rect(surf, black, (x, y, size * 0.5, size * 0.4), 0)
        draw.rect(surf, black, (x, y, size*0.5, size*0.4), 2)
        # print_text(values[0], (x+0.045*size, y+0.05*size), black)
        print_text(tile_emoji[tile], (x + 0.045 * size, y + 0.05 * size), tile_color[tile], emoji=True)

        # Die
        die = values[1]
        nx = x+0.1*size
        ny = y+size*0.5
        draw.rect(surf, die_color[die], (nx, ny, size*0.4, size*0.4), 0)
        print_text(die, (nx+0.1*size, ny+0.05*size), white)

        # Gradient bar
        def draw_gradient(c1, c2, x, y, width, height):
            act_h = 0
            step = (c2-c1)/height
            draw.rect(surf, black, (x-1, y-height, width+3, height+3), 0)
            while (act_h/height - proportion) < 0.01 and act_h <= height:
                act_c = c1 + step*act_h
                act_y = y - act_h
                draw.line(surf, act_c, (x, act_y), (x + width, act_y), 1)
                act_h += 1

        proportion = values[3]
        nx = x + size*0.6
        ny = y + size*0.9
        draw_gradient(np.array(pale_turquoise), np.array(orange_red), nx, ny, size*0.3, size*0.7)

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
            surf.blit(text[0], (x+10,y+10))

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
    if not os.path.exists(directory):
        os.makedirs(directory)
    image.save(surf, os.path.join(directory, filename))


if __name__ == "__main__":
    seed(456)
    boards = [
        Board(defaultdict(list), circling=False), Board(defaultdict(list), circling=False),
              Board({"trap1": [10], "trap2": [4, 5, 8], "trap3":[]}, circling=False),
              Board({"trap1": [9, 13], "trap2": [8, 12, 4], "trap3": [1, 10, 11]}, circling=True)
              ]
    names = ["p1", "p2", "p3", "p4"]
    for board, name in zip(boards, names):
        mdp_expec, mdp_dice = markovDecision(board)
        # mdp_dice = find_strategy(board, n_games=5000, very_intelligent=True)

        # mdp_expec = play_strategy(board, lambda x: mdp_dice[x]-1, n_games=10000)
        #
        # ag_dice = find_strategy(board, n_games=1000, very_intelligent=False)
        # ag_expec = play_strategy(board, lambda x: ag_dice[x], n_games=10000)
        #
        # rand_expec = play_strategy(board, lambda x: randrange(3), n_games=10000)

        draw_board([x.type for x in board.board], mdp_dice, mdp_expec, mdp_expec[0], name, "mdp" + ".png")
        # draw_board([x.type for x in board.board], mdp_dice, mdp_expec, ag_expec[0], name,"mdp"+".png")
        # draw_board([x.type for x in board.board], [d+1 for d in ag_dice], ag_expec, ag_expec[0], name, "ag" + ".png")
        # draw_board([x.type for x in board.board], [0]*15, rand_expec, ag_expec[0], name, "rand" + ".png")

    # ' & '.join([str(x) for x in a])+'\\'

    # board = [x.type for x in Board({'trap1': [1, 10], 'trap2': [3, 5, 7]}).board]
    # dice = [2, 1, 3, 3, 2, 3, 3, 3, 2, 1, 1, 3, 2, 1, 1]
    # expec = [11.698915143254899, 11.032249954001127, 9.032252820017574, 9.701159919483432, 8.43010450515775, 7.401431295786926, 5.643368090782532, 4.232526126289438, 2.5, 2.0, 4.833333333333332, 2.833333333333333, 2.5, 2.0, 0]
    # ref_expec = expec[0]
    # draw_board(board, dice, expec, ref_expec, "board.png")
