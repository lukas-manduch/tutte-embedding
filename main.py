from pprint import pprint
import numpy as np
from numpy.linalg import solve
import matplotlib as plt
import matplotlib.lines
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import networkx as nx

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

POINTS = list()

def add_line(x1, x2):
    x ,y = zip(x1, x2)
    print("Line: " + str(x) + "  " + str(y))
    global POINTS
    POINTS.append(x1)
    POINTS.append(x2)
    plt.plot(x, y)
    #l = Line2D(x, y)
    #plt.gca().add_line(l)

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

def solve_for(dimension: int, inp: dict, given:dict):
    equations = np.zeros((len(inp), len(inp)))
    results = np.zeros(len(inp))
    #index = list(zip(inp.keys(), range(len(inp))))
    index = list(inp.keys())
    ###
    for i in range(len(index)):
        value = index[i]
        if value in given:
            results[i] = given[value][dimension]
            equations[i][i] = 1
        else:
            for neighbour in inp[value]:
                equations[i][index.index(neighbour)] = 1*(1/len(inp[value]))
                equations[i][i] = -1

    #pprint(equations)
    res = solve(equations, results)
    return res


def tutte(inp: dict, circle_vertices: list):
    positions = dict(zip(circle_vertices, create_circle(len(circle_vertices))))
    pprint(positions)
    np.zeros((1, 3))
    len(inp)
    vals = list(zip(solve_for(0, inp, positions),
                    solve_for(1, inp, positions)))
    return dict(zip(inp.keys(), vals))


if __name__ == "__main__":
    pprint(gr1)
    results = tutte(gr2, [1, 2, 3, 7])
    pprint(results)
    draw_lines(results, gr2)
    annotate(results)

    plt.show()

pprint(sorted(POINTS))
