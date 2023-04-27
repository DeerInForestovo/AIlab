import copy

import numpy as np
import sys
import time
import pandas as pd


TEST_MODE = 1


class CarpProblem:
    def __init__(self, argv):
        # input
        self.begin_time = time.time()
        file = open(argv[0], 'r')
        self.time_limit = self.begin_time + (int(argv[2]) - 0.233333)
        self.random_seed = argv[4]
        read_table = dict()
        for i in range(8):
            next_line = file.readline()
            name, value = next_line.split(':')
            read_table[name] = value
        self.req_edge = int(read_table['REQUIRED EDGES '])
        self.free_edge = int(read_table['NON-REQUIRED EDGES '])
        self.node = int(read_table['VERTICES '])
        self.car = int(read_table['VEHICLES '])
        self.capacity = int(read_table['CAPACITY '])
        self.depot = int(read_table['DEPOT '])
        self.test_name = read_table['NAME ']
        self.total_cost = int(read_table['TOTAL COST OF REQUIRED EDGES '])
        self.edge = self.req_edge + self.free_edge
        read_edges = pd.read_csv(file, sep="\s+", skiprows=1, skipfooter=1, names=['1', '2', '3', '4']
                                 , engine='python').values
        file.close()
        self.distance = [[self.total_cost for _ in range(self.node)] for _ in range(self.node)]
        for i in range(self.node):
            self.distance[i][i] = 0
        self.task = []
        for i in range(self.edge):
            u, v, c, d = read_edges[i]
            u -= 1
            v -= 1
            self.distance[u][v] = self.distance[v][u] = c
            if d:
                self.task.append((u, v, d))
        assert len(self.task) == self.req_edge

        # Floyd
        for k in range(self.node):
            for i in range(self.node):
                for j in range(self.node):
                    self.distance[i][j] = min(self.distance[i][j], self.distance[i][k] + self.distance[k][j])
        """
            parameters:
            self.time_limit: time limit
                (Make sure time.time() < self.time_limit)
            self.random_seed: random seed
            self.req_edge: number of required edges
            self.free_edge: number of not required edges
            self.node: number of vertices
            self.car: number of vehicles
            self.capacity: vehicles' capacity
            self.task: list of (u, v, d) representing an required edge (u, v) with demand d
                (we can assert len(self.task) == self.req_edge)
            self.distance: distance[i][j] representing distance between node i and node j
        """

    """
        5 strategies:
        1) maximize the distance from the task to the depot;
        2) minimize the distance from the task to the depot;
        3) maximize the term dem(t)/sc(t), where dem(t) and sc(t) are demand and serving cost of task t, respectively;
        4) minimize the term dem(t)/sc(t);
        5) use rule 1) if the vehicle is less than half- full, otherwise use rule 2) 
    """
    def strategy(self, cap_now, task_now, pos_now, strategy_id):
        if strategy_id == 0:
            best_pos = -1
            max_dist = -1
            for i, this_task in enumerate(task_now):
                u, v, d = this_task
                if cap_now + d <= self.capacity and (max_dist < self.distance[self.depot][u] or best_pos == -1):
                    max_dist = self.distance[self.depot][u]
                    best_pos = i
            return best_pos
        elif strategy_id == 1:
            best_pos = -1
            min_dist = -1
            for i, this_task in enumerate(task_now):
                u, v, d = this_task
                if cap_now + d <= self.capacity and (min_dist > self.distance[self.depot][u] or best_pos == -1):
                    min_dist = self.distance[self.depot][u]
                    best_pos = i
            return best_pos
        elif strategy_id == 2:
            best_pos = -1
            max_ratio = -1
            for i, this_task in enumerate(task_now):
                u, v, d = this_task
                sc = self.distance[pos_now][u] + self.distance[u][v]
                if cap_now + d <= self.capacity and (max_ratio < d / sc or best_pos == -1):
                    max_ratio = d / sc
                    best_pos = i
            return best_pos
        elif strategy_id == 3:
            best_pos = -1
            min_ratio = -1
            for i, this_task in enumerate(task_now):
                u, v, d = this_task
                sc = self.distance[pos_now][u] + self.distance[u][v]
                if cap_now + d <= self.capacity and (min_ratio > d / sc or best_pos == -1):
                    min_ratio = d / sc
                    best_pos = i
            return best_pos
        elif strategy_id == 4:
            return self.strategy(cap_now, task_now, pos_now, int(cap_now * 2 < self.capacity))
        else:
            return -1

    """
        individual = (list of routes, cost)
        route = [task1, task2, ...]
        len(list of routes) = self.car
    """
    def init_population(self, pop_size):
        population = []
        STRATEGY_NUM = 5
        for i in range(pop_size):
            routes = []
            cost = 0
            task_now = copy.deepcopy(self.task)
            for j in range(self.req_edge):
                if np.random.randint(2):
                    task_now[j] = (task_now[j][1], task_now[j][0], task_now[j][2])
            for j in range(self.car):
                route = []
                cap_now = 0
                pos_now = self.depot
                while True:
                    best_pos = self.strategy(cap_now, task_now, pos_now, i % STRATEGY_NUM)
                    if best_pos == -1:
                        cost += self.distance[self.depot][route[-1][1]]  # the end point of the last task
                        break
                    best_task = task_now.pop(best_pos)
                    route.append(best_task)
                    cost += self.distance[pos_now][best_task[0]] + self.distance[best_task[0]][best_task[1]]
                    cap_now += best_task[2]
                    pos_now = best_task[1]
                routes.append(route)
            population.append((routes, cost))
        return population

    def main(self):  # solve the problem
        # init population
        pop_size = max(1000, min(10, int(1000000 / self.node)))  # need a better number here
        population = self.init_population(pop_size)
        if TEST_MODE:
            print("Time test: init_population -> %f s" % (time.time() - self.begin_time))


if __name__ == "__main__":
    CarpProblem(sys.argv[1:]).main()
    """
        example test:
        python CARP_solver.py sample.dat -t 5 -s 0
    """

