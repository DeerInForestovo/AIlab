import copy

import numpy as np
import sys
import time
import pandas as pd
import random

TEST_MODE = 0


class CarpProblem:
    def __init__(self, argv):
        # input
        self.begin_time = time.time()
        file = open(argv[0], 'r')
        self.time_limit = self.begin_time + (int(argv[2]) - 0.1)
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
        self.depot = int(read_table['DEPOT ']) - 1
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
                self.task.append((int(u), int(v), int(d)))
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
        best_pos = -1
        if strategy_id == 0:
            max_dist = -1
            for i, this_task in enumerate(task_now):
                u, v, d = this_task
                if cap_now + d <= self.capacity and (max_dist < self.distance[self.depot][u] or best_pos == -1):
                    max_dist = self.distance[self.depot][u]
                    best_pos = i
        if strategy_id == 1:
            min_dist = -1
            for i, this_task in enumerate(task_now):
                u, v, d = this_task
                if cap_now + d <= self.capacity and (min_dist > self.distance[self.depot][u] or best_pos == -1):
                    min_dist = self.distance[self.depot][u]
                    best_pos = i
        if strategy_id == 2:
            max_ratio = -1
            for i, this_task in enumerate(task_now):
                u, v, d = this_task
                sc = self.distance[pos_now][u] + self.distance[u][v]
                if cap_now + d <= self.capacity and (max_ratio < d / sc or best_pos == -1):
                    max_ratio = d / sc
                    best_pos = i
        if strategy_id == 3:
            min_ratio = -1
            for i, this_task in enumerate(task_now):
                u, v, d = this_task
                sc = self.distance[pos_now][u] + self.distance[u][v]
                if cap_now + d <= self.capacity and (min_ratio > d / sc or best_pos == -1):
                    min_ratio = d / sc
                    best_pos = i
        if strategy_id == 4:
            best_pos = self.strategy(cap_now, task_now, pos_now, int(cap_now * 2 < self.capacity))
        return best_pos

    def calc_cost(self, individual):  # only used for testing
        cost = 0
        for route in individual[0]:
            len_route = len(route)
            for i in range(len_route):
                if i:
                    cost += self.distance[route[i - 1][1]][route[i][0]]
                cost += self.distance[route[i][0]][route[i][1]]
            cost += self.distance[self.depot][route[0][0]] + self.distance[route[-1][1]][self.depot]
        return cost

    """
        5 mutations:
        flip, single insertion, double insertion, swap, 2-opt
    """

    def cost_right(self, routes, a, b):
        return self.distance[routes[a][b][1]][routes[a][(b + 1) % len(routes[a])][0]]

    def cost_left(self, routes, a, b):
        return self.distance[routes[a][b - 1][1]][routes[a][b][0]]

    def cost_both(self, routes, a, b):
        return self.cost_left(routes, a, b) + self.cost_right(routes, a, b)

    def random_task(self, routes):
        random_id = np.random.randint(self.req_edge)
        for i, route in enumerate(routes):
            if random_id >= len(route) - 1:
                random_id -= len(route) - 1
            else:
                return i, random_id + 1

    def route_test_failed(self, route):
        demand_sum = 0
        for task in route:
            demand_sum += task[2]
        return demand_sum > self.capacity

    def mutation(self, individual, mutation_id):
        routes = copy.deepcopy(individual[0])
        cost = individual[1]
        if mutation_id == 0:  # flip
            a, b = self.random_task(routes)
            cost -= self.cost_both(routes, a, b)
            routes[a][b] = (routes[a][b][1], routes[a][b][0], routes[a][b][2])
            cost += self.cost_both(routes, a, b)
        if mutation_id == 1:  # double insertion
            a, b = self.random_task(routes)
            c, d = self.random_task(routes)
            if b + 1 == len(routes[a]) or a == c:
                mutation_id = 2  # the next task is init_task or inserted to wrong positions
            else:
                cost -= self.cost_left(routes, a, b) +\
                        self.cost_right(routes, a, b + 1) +\
                        self.cost_right(routes, c, d)
                task_a1 = routes[a].pop(b)
                task_a2 = routes[a].pop(b)
                routes[c].insert(d + 1, task_a2)
                routes[c].insert(d + 1, task_a1)
                cost += self.cost_right(routes, a, b - 1) +\
                        self.cost_left(routes, c, d + 1) +\
                        self.cost_right(routes, c, d + 2)
                if self.route_test_failed(routes[a]) or self.route_test_failed(routes[c]):
                    return None
        if mutation_id == 2:  # single insertion
            a, b = self.random_task(routes)
            c, d = self.random_task(routes)
            if b + 1 == len(routes[a]) or a == c:
                mutation_id = 3  # the next task is init_task or inserted to wrong positions
            else:
                cost -= self.cost_both(routes, a, b) +\
                        self.cost_right(routes, c, d)
                task_a = routes[a].pop(b)
                routes[c].insert(d + 1, task_a)
                cost += self.cost_right(routes, a, b - 1) +\
                        self.cost_both(routes, c, d + 1)
                if self.route_test_failed(routes[a]) or self.route_test_failed(routes[c]):
                    return None
        if mutation_id == 3:  # swap
            a, b = self.random_task(routes)
            c, d = self.random_task(routes)
            if a != c:
                cost -= self.cost_both(routes, a, b) +\
                        self.cost_both(routes, c, d)
                routes[a][b], routes[c][d] = routes[c][d], routes[a][b]
                cost += self.cost_both(routes, a, b) +\
                        self.cost_both(routes, c, d)
                if self.route_test_failed(routes[a]) or self.route_test_failed(routes[c]):
                    return None
        if mutation_id == 4:
            pass
        return routes, cost

    """
        individual = (list of routes, cost)
            len(list of routes) = self.card
        route = [init_task, task1, task2, ...]
            (recall task = (u, v, d), init_task = (self.depot, self.depot, 0))
    """

    def init_population(self, pop_size):
        population = []
        STRATEGY_NUM = 5
        for i in range(pop_size):
            routes = []
            cost = 0
            task_now = copy.deepcopy(self.task)
            random.shuffle(task_now)
            for j in range(self.req_edge):
                if np.random.randint(2):
                    task_now[j] = (task_now[j][1], task_now[j][0], task_now[j][2])
            for j in range(self.car):
                route = [(self.depot, self.depot, 0)]
                cap_now = 0
                pos_now = self.depot
                while True:
                    best_pos = self.strategy(cap_now, task_now, pos_now, i % STRATEGY_NUM)
                    if best_pos == -1:
                        cost += self.distance[route[-1][1]][self.depot]  # the end point of the last task
                        break
                    best_task = task_now.pop(best_pos)
                    route.append(best_task)
                    cost += self.distance[pos_now][best_task[0]] + self.distance[best_task[0]][best_task[1]]
                    cap_now += best_task[2]
                    pos_now = best_task[1]
                routes.append(route)
            if len(task_now) == 0:
                population.append((routes, cost))
        return population

    def best_n_individual(self, population, n):  # fitness = -cost
        return sorted(population, key=lambda x: x[1])[:n]

    def main(self):  # solve the problem
        # init population
        pop_size = max(1000, min(10, int(1000000 / self.node)))  # need a better number here
        population = self.init_population(pop_size)
        if TEST_MODE:
            print("init_population in %f s" % (time.time() - self.begin_time))
        phase_num = 0
        phase_time = 0
        while time.time() + phase_time < self.time_limit:
            phase_begin_time = time.time()
            if TEST_MODE:
                for individual in population:
                    if self.calc_cost(individual) != individual[1]:
                        print("wrong cost")
                        print(individual)
            # generate new population
            new_population = [] + population
            for individual in population:
                MUTATION_NUM = 5
                for i in range(MUTATION_NUM):
                    new_individual = self.mutation(individual, i)
                    if new_individual is not None:
                        new_population.append(new_individual)
            # replace
            population = self.best_n_individual(new_population, pop_size)
            phase_time = time.time() - phase_begin_time
            phase_num += 1
            if TEST_MODE:
                print("phase %d, time %f, best time %d\n" % (phase_num, phase_time, population[0][1]))
        # output
        best_individual = self.best_n_individual(population, 1)[0]
        plan_output = ''
        for route in best_individual[0]:
            if len(plan_output):
                plan_output += ','
            plan_output += '0'
            for task in route[1:]:
                plan_output += str(',(%d,%d)' % (task[0] + 1, task[1] + 1))
            plan_output += ',0'
        print('s', plan_output)
        print('q', best_individual[1])


if __name__ == "__main__":
    CarpProblem(sys.argv[1:]).main()
    """
        example test:
        python CARP_solver.py sample0.dat -t 5 -s 0
    """
