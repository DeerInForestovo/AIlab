import copy
import numpy as np
import sys
import time

STRATEGY_NUM = 5
MUTATION_NUM = 7

TEST_MODE = 0


def read_int(string, begin):  # very stupid function
    p = 0
    while not str(string[begin]).isdigit():
        begin += 1
    while str(string[begin]).isdigit():
        p = p * 10 + int(string[begin])
        begin += 1
    return p, begin


def best_n_individual(population, n):  # fitness = -cost
    return sorted(population, key=lambda x: x[1])[:n]


def total_length(individual):  # only for testing
    length = 0
    for route in individual[0]:
        length += len(route)
    return length


class CarpProblem:
    def __init__(self, argv):
        # input
        self.begin_time = time.time()
        file = open(argv[0], 'r')
        self.time_limit = self.begin_time + (int(argv[2]) - 2)
        self.random_seed = argv[4]
        read_table = dict()
        for i in range(8):
            next_line = file.readline()
            name, value = next_line.split(':')
            read_table[name] = value
        self.req_edge = int(read_table['REQUIRED EDGES '])
        self.free_edge = int(read_table['NON-REQUIRED EDGES '])
        self.node = int(read_table['VERTICES '])
        # self.car = int(read_table['VEHICLES '])
        self.car = self.req_edge  # change the number of vehicles to +INF due to the teacher's requirement
        self.capacity = int(read_table['CAPACITY '])
        self.depot = int(read_table['DEPOT ']) - 1
        self.test_name = read_table['NAME ']
        self.total_req_cost = int(read_table['TOTAL COST OF REQUIRED EDGES '])
        self.total_cost = 0
        self.edge = self.req_edge + self.free_edge
        file.readline()
        self.distance = [[-1 for _ in range(self.node)] for _ in range(self.node)]
        for i in range(self.node):
            self.distance[i][i] = 0
        self.task = [(self.depot, self.depot, 0)]
        for i in range(self.edge):
            # very stupid and ugly input
            string = file.readline() + '\0'
            u, begin = read_int(string, 0)
            v, begin = read_int(string, begin)
            c, begin = read_int(string, begin)
            d, _ = read_int(string, begin)
            u -= 1
            v -= 1
            self.total_cost += c
            if TEST_MODE:
                assert 0 <= u < self.node and 0 <= v < self.node
            self.distance[u][v] = self.distance[v][u] = c
            if d:
                self.task.append((int(u), int(v), int(d)))
        if TEST_MODE:
            assert len(self.task) == self.req_edge + 1
        file.close()

        # Floyd
        for i in range(self.node):
            for j in range(self.node):
                if self.distance[i][j] == -1:
                    self.distance[i][j] = self.total_cost
        for k in range(self.node):
            for i in range(self.node):
                for j in range(self.node):
                    self.distance[i][j] = min(self.distance[i][j], self.distance[i][k] + self.distance[k][j])
        if TEST_MODE:
            print("init time %f s" % (time.time() - self.begin_time))

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

    def get_task(self, id):
        return self.task[id] if id >= 0 else (self.task[-id][1], self.task[-id][0], self.task[-id][2])

    def strategy(self, cap_now, task_now, pos_now, strategy_id):
        best_pos = -1
        if strategy_id == 0:
            max_dist = -1
            for i, this_task in enumerate(task_now):
                u, v, d = self.get_task(this_task)
                if cap_now + d <= self.capacity and (max_dist < self.distance[self.depot][u] or best_pos == -1):
                    max_dist = self.distance[self.depot][u]
                    best_pos = i
        if strategy_id == 1:
            min_dist = -1
            for i, this_task in enumerate(task_now):
                u, v, d = self.get_task(this_task)
                if cap_now + d <= self.capacity and (min_dist > self.distance[self.depot][u] or best_pos == -1):
                    min_dist = self.distance[self.depot][u]
                    best_pos = i
        if strategy_id == 2:
            max_ratio = -1
            for i, this_task in enumerate(task_now):
                u, v, d = self.get_task(this_task)
                sc = self.distance[pos_now][u] + self.distance[u][v]
                if cap_now + d <= self.capacity and (max_ratio < d / sc or best_pos == -1):
                    max_ratio = d / sc
                    best_pos = i
        if strategy_id == 3:
            min_ratio = -1
            for i, this_task in enumerate(task_now):
                u, v, d = self.get_task(this_task)
                sc = self.distance[pos_now][u] + self.distance[u][v]
                if cap_now + d <= self.capacity and (min_ratio > d / sc or best_pos == -1):
                    min_ratio = d / sc
                    best_pos = i
        if strategy_id == 4:
            best_pos = self.strategy(cap_now, task_now, pos_now, int(cap_now * 2 < self.capacity))
        return best_pos

    def calc_cost(self, routes):
        cost = 0
        for route in routes:
            for i in range(len(route) - 1):
                # DON'T DELETE THIS COMMENT
                # cost += self.distance[route[i - 1][1]][route[i][0]] +\
                #         self.distance[route[i][0]][route[i][1]]

                # Before constant improvement
                # cost += self.distance[route[i - 1][1]][route[i][0]]

                _, last_v, _ = self.get_task(route[i])
                next_u, _, _ = self.get_task(route[i+1])
                cost += self.distance[last_v][next_u]
        # return cost
        return cost + self.total_req_cost
        # Commented method is WRONG!!! Demanded edge (u, v) may be longer than distance[u][v].

    """
        5 mutations:
        flip, single insertion, double insertion, swap, 2-opt
    """

    def cost_right(self, routes, a, b):
        _, last_v, _ = self.get_task(routes[a][b])
        next_u, _, _ = self.get_task(routes[a][b+1])
        return self.distance[last_v][next_u]

    def cost_left(self, routes, a, b):
        _, last_v, _ = self.get_task(routes[a][b-1])
        next_u, _, _ = self.get_task(routes[a][b])
        return self.distance[last_v][next_u]
        # one of this and cost_right is enough, but I don't want to change anymore...

    def cost_both(self, routes, a, b):
        return self.cost_left(routes, a, b) + self.cost_right(routes, a, b)

    def random_task(self, routes):
        random_id = np.random.randint(self.req_edge)
        for i, route in enumerate(routes):
            if random_id >= len(route) - 2:
                random_id -= len(route) - 2
            else:
                return i, random_id + 1

    def route_test_failed(self, route):
        demand_sum = 0
        for id in route:
            demand_sum += self.task[abs(id)][2]
        return demand_sum > self.capacity

    def mutation(self, individual, mutation_id):
        routes = copy.deepcopy(individual[0])
        cost = individual[1]
        if mutation_id == 0:  # flip
            a, b = self.random_task(routes)
            cost -= self.cost_both(routes, a, b)
            routes[a][b] = -routes[a][b]
            cost += self.cost_both(routes, a, b)
        if mutation_id == 1:  # double insertion
            a, b = self.random_task(routes)
            c, d = self.random_task(routes)
            if b + 2 == len(routes[a]) or a == c:
                mutation_id = 2  # the next task is init_task or inserted to wrong positions
            else:
                cost -= self.cost_left(routes, a, b) + \
                        self.cost_right(routes, a, b+1) + \
                        self.cost_right(routes, c, d)
                task_a1 = routes[a].pop(b)
                task_a2 = routes[a].pop(b)
                routes[c].insert(d + 1, task_a2)
                routes[c].insert(d + 1, task_a1)
                cost += self.cost_right(routes, a, b-1) + \
                        self.cost_left(routes, c, d+1) + \
                        self.cost_right(routes, c, d+2)
                if self.route_test_failed(routes[a]) or self.route_test_failed(routes[c]):
                    return None
        if mutation_id == 2:  # single insertion
            a, b = self.random_task(routes)
            c, d = self.random_task(routes)
            if b + 2 == len(routes[a]) or a == c:
                mutation_id = 3  # the next task is init_task or inserted to wrong positions
            else:
                cost -= self.cost_both(routes, a, b) + \
                        self.cost_right(routes, c, d)
                task_a = routes[a].pop(b)
                routes[c].insert(d + 1, task_a)
                cost += self.cost_right(routes, a, b-1) + \
                        self.cost_both(routes, c, d+1)
                if self.route_test_failed(routes[a]) or self.route_test_failed(routes[c]):
                    return None
        if mutation_id == 3:  # swap
            a, b = self.random_task(routes)
            c, d = self.random_task(routes)
            if a == c:
                mutation_id = 4
            else:
                cost -= self.cost_both(routes, a, b) + \
                        self.cost_both(routes, c, d)
                routes[a][b], routes[c][d] = routes[c][d], routes[a][b]
                cost += self.cost_both(routes, a, b) + \
                        self.cost_both(routes, c, d)
                if self.route_test_failed(routes[a]) or self.route_test_failed(routes[c]):
                    return None
        if mutation_id == 4:  # 2-opt for double route (plan 1)
            a, b = self.random_task(routes)
            c, d = self.random_task(routes)
            if a == c:
                mutation_id = 5
            else:
                cost -= self.cost_left(routes, a, b) + \
                        self.cost_left(routes, c, d)
                routes[a], routes[c] = routes[a][:b] + routes[c][d:], routes[c][:d] + routes[a][b:]
                cost += self.cost_right(routes, a, b-1) + \
                        self.cost_right(routes, c, d-1)
                if self.route_test_failed(routes[a]) or self.route_test_failed(routes[c]):
                    return None
        if mutation_id == 5:  # 2-opt for double route (plan 2)
            a, b = self.random_task(routes)
            c, d = self.random_task(routes)
            if a == c:
                mutation_id = 6
            else:
                cost -= self.cost_left(routes, a, b) + \
                        self.cost_right(routes, c, d)
                len_a = len(routes[a])
                for i in range(b, len_a-1):
                    routes[a][i] = -routes[a][i]
                for i in range(1, d+1):
                    routes[c][i] = -routes[c][i]
                routes[a], routes[c] = routes[a][:b] + routes[c][d::-1], routes[a][:b-1:-1] + routes[c][d+1:]
                cost += self.cost_left(routes, a, b) + \
                        self.cost_right(routes, c, len_a-b-1)
                if self.route_test_failed(routes[a]) or self.route_test_failed(routes[c]):
                    return None
        if mutation_id == 6:  # 2-opt for single route (segment reverse)
            a, b = self.random_task(routes)
            c = np.random.randint(len(routes[a]) - 2) + 1
            b, c = min(b, c), max(b, c)
            cost -= self.cost_left(routes, a, b) + \
                    self.cost_right(routes, a, c)
            routes[a][b:c+1] = routes[a][c:b-1:-1]  # segment reverse
            for i in range(b, c + 1):
                routes[a][i] = -routes[a][i]
            cost += self.cost_left(routes, a, b) + \
                    self.cost_right(routes, a, c)
        return routes, cost

    """
        individual = (list of routes, cost)
            len(list of routes) = self.car
        ~~route = [init_task, task1, task2, ...]~~
            ~~(recall task = (u, v, d), init_task = (self.depot, self.depot, 0))~~
        route = [0, task1_id, task2_id, ..., 0]
            id > 0 -> (u, v, d), id < 0 -> (v, u, d) 
    """

    def init_population(self, pop_size):
        population = []
        for i in range(pop_size):
            routes = []
            task_now = [i + 1 for i in range(self.req_edge)]
            for j in range(self.req_edge):
                if np.random.randint(2):
                    task_now[j] = -task_now[j]
            for j in range(self.car):
                route = [0]
                cap_now = 0
                pos_now = self.depot
                while True:
                    best_pos = self.strategy(cap_now, task_now, pos_now, i % STRATEGY_NUM)
                    if best_pos == -1:
                        break
                    best_task = task_now.pop(best_pos)
                    route.append(best_task)
                    u, v, d = self.get_task(best_task)
                    cap_now += d
                    pos_now = v
                route.append(0)
                routes.append(route)
            if len(task_now):
                population.append((routes, self.calc_cost(routes)))
        return population

    def main(self):  # solve the problem
        phase_num = 0
        phase_time = 0
        init_time = 0
        round_cnt = 0
        best_individual = None
        while time.time() + init_time < self.time_limit:
            # init population
            init_begin_time = time.time()
            pop_size = 100
            population = self.init_population(pop_size)  # init population
            init_time = time.time() - init_begin_time
            cost_history = []
            if TEST_MODE:
                round_cnt += 1
                print("round %d begins here." % round_cnt)
                print("init_population in %f s" % init_time)

            # evolution
            while time.time() + phase_time < self.time_limit:
                phase_begin_time = time.time()
                if TEST_MODE:
                    for individual in population:
                        if self.calc_cost(individual[0]) != individual[1]:
                            print("wrong cost")
                            print(individual)
                        if total_length(individual) != self.car + self.car + self.req_edge:
                            print("wrong task number")
                            print(individual)

                # generate new population
                new_population = [] + population
                for individual in population:
                    for i in range(MUTATION_NUM):
                        new_individual = self.mutation(individual, i)
                        if new_individual is not None:
                            new_population.append(new_individual)

                # replace
                population = best_n_individual(new_population, pop_size)
                best_cost = int(population[0][1])
                cost_history.append(best_cost)

                # phase time calc
                phase_time = time.time() - phase_begin_time
                phase_num += 1
                if TEST_MODE:
                    print("phase %d, time %f, best cost %d" % (phase_num, phase_time, population[0][1]))

                # if evolution ends, generate a new population
                if len(cost_history) > pop_size//2 and best_cost == cost_history[-(pop_size//2)]:
                    if TEST_MODE:
                        print("cost history:")
                        print(cost_history)
                        print("go to next population")
                    break

            # update the answer
            best_individual_this_round = population[0]
            if best_individual is None or best_individual[1] > best_individual_this_round[1]:
                best_individual = best_individual_this_round

        # output
        plan_output = ''
        for route in best_individual[0]:
            if len(route) == 2:
                continue
            if len(plan_output):
                plan_output += ','
            plan_output += '0'
            for task in route:
                if task:
                    u, v, _ = self.get_task(task)
                    plan_output += str(',(%d,%d)' % (u + 1, v + 1))
            plan_output += ',0'
        print('s', plan_output)
        print('q', best_individual[1])


if __name__ == "__main__":
    CarpProblem(sys.argv[1:]).main()
    """
        example test:
        python CARP_solver.py sample0.dat -t 5 -s 0
    """

