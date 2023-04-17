import numpy as np
import sys
import time
import pandas as pd


def main(argv):
    """
        输入部分写的是依托答辩
        如果你正在阅读这份代码，请一定不要参考这个输入
    """
    file = open(argv[0], 'r')
    run_time = argv[2]
    random_seed = argv[4]
    read_table = dict()
    for i in range(8):
        next_line = file.readline()
        name, value = next_line.split(':')
        read_table[name] = value
    req_edge = int(read_table['REQUIRED EDGES '])
    free_edge = int(read_table['NON-REQUIRED EDGES '])
    node = int(read_table['VERTICES '])
    car = int(read_table['VEHICLES '])
    capacity = int(read_table['CAPACITY '])
    depot = int(read_table['DEPOT '])
    test_name = read_table['NAME ']
    total_cost = int(read_table['TOTAL COST OF REQUIRED EDGES '])
    edge = req_edge + free_edge
    read_edges = pd.read_csv(file, sep="\s+", skiprows=1, skipfooter=1, names=['1', '2', '3', '4']
                             , engine='python').values
    distance = [[total_cost for _ in range(node)] for _ in range(node)]
    for i in range(node):
        distance[i][i] = 0
    task = []
    for i in range(edge):
        u, v, c, d = read_edges[i]
        u -= 1
        v -= 1
        distance[u][v] = distance[v][u] = c
        if d:
            task.append((u, v, d))
    assert len(task) == req_edge

    # Floyd
    for k in range(node):
        for i in range(node):
            for j in range(node):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])

    # GA


if __name__ == "__main__":
    main(sys.argv[1:])

