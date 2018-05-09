from pprint import pprint
import numpy as np
from numpy.linalg import solve
import matplotlib as plt
import matplotlib.lines
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import random


gr1 = {
    1: [2, 4, 5],
    2: [1, 3, 5, 6],
    3: [2, 4, 6],
    4: [1, 5, 3],
    5: [1, 2, 4, 6],
    6: [2, 3, 4, 5]
}

gr2 = {
    1 : [2, 6],
    2: [1, 3],
    3: [2, 4, 7],
    4: [3, 5, 8, 9],
    5: [4, 6, 8],
    6: [1, 5, 9],
    7: [1, 3],
    8: [4, 5, 6],
    9: [3, 6]
}

gr3 = { # 20 sided
    2: [6, 12, 17, 37, 27],
    6: [2, 12, 17, 31, 33],
   12: [2, 6, 27, 46, 31],
   17: [2, 6, 33, 51, 37],
   27: [2, 12, 37, 54, 46],
   31: [6, 12, 46, 58, 33],
   33: [6, 17, 31, 58, 51],
   37: [2, 17, 27, 51, 54],
   46: [12, 27, 31, 54, 58],
   51: [17, 33, 37, 54, 58],
   54: [27, 37, 46, 51, 58],
   58: [31, 33, 46, 51, 54]
   }

gr4 = { # 12 sided
    4: [8, 11, 13],
    8: [4, 16, 18],
    11: [4, 20, 28],
    13: [4, 30, 23],
    16: [8, 23, 34],
    18: [8, 36, 20],
    20: [11, 18, 38],
    23: [13, 16, 41],
    28: [11, 30, 45],
    30: [13, 28, 47],
    34: [16, 50, 36],
    36: [18, 34, 52],
    38: [20, 45, 52],
    41: [23, 47, 50],
    45: [28, 38, 56],
    47: [30, 41, 56],
    50: [34, 41, 60],
    52: [36, 38, 60],
    56: [45, 47, 60],
    60: [50, 52, 56]
}

def add_line(x1, x2):
    x ,y = zip(x1, x2)
    print("Line: " + str(x) + "  " + str(y))
    plt.plot(x, y)

def create_circle(num_vertex):
    g = nx.Graph()
    for i in range(num_vertex):
        g.add_edge(i, (i+1)%num_vertex)
    d = nx.spectral_layout(g)
    return list(d.values())

def draw_lines(positions, inp):
    for vertex, neighbours in inp.items():
        for neighbour in neighbours:
            print("From {} to {}".format(vertex, neighbour))
            add_line([positions[vertex][0], positions[vertex][1]],
                     [positions[neighbour][0], positions[neighbour][1]])

def annotate(positions):
    for label, point in positions.items():
        plt.gca().annotate(label, point)

def solve_for(dimension: int, inp: dict, given: dict, weights = defaultdict(lambda: 1)):
    equations = np.zeros((len(inp), len(inp)))
    results = np.zeros(len(inp))
    index = list(inp.keys())
    ###
    for i in range(len(index)):
        value = index[i]
        if value in given:
            results[i] = given[value][dimension]
            equations[i][i] = 1
        else:
            sum_neighbours = sum([weights[x] for x in inp[value]])
            for neighbour in inp[value]:
                equations[i][index.index(neighbour)] = weights[neighbour]*(1/sum_neighbours)
                equations[i][i] = -1
    res = solve(equations, results)
    return res


def tutte(inp: dict, circle_vertices: list):
    positions = dict(zip(circle_vertices, create_circle(len(circle_vertices))))
    pprint(positions)
    np.zeros((1, 3))
    weights = defaultdict(lambda: 8)
    for key in inp:
        weights[key] = random.random()
    vals = list(zip(solve_for(0, inp, positions, weights),
                    solve_for(1, inp, positions, weights)))
    return dict(zip(inp.keys(), vals))


if __name__ == "__main__":
    graph = gr4
    circle = [4, 8,  16,  23, 13]
    results = tutte(graph, circle)
    draw_lines(results, graph)
    pprint(results)
    annotate(results)

    plt.show()
