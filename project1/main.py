import copy

import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
Infinity = 1000000
local_test = 0
dx = [-1, -1, -1, 0, 0, 1, 1, 1]
dy = [-1, 0, 1, -1, 1, -1, 0, 1]
dp = [-9, -8, -7, -1, 1, 7, 8, 9]
board_worth = [
    -10, 3, -3, 0, 0, -3, 3, -10,
    3, -1, -3, 1, 1, -3, -1, 3,
    -3, -3, 1, 1, 1, 1, -3, -3,
    0, 1, 1, 10, 10, 1, 1, 0,
    0, 1, 1, 10, 10, 1, 1, 0,
    -3, -3, 1, 1, 1, 1, -3, -3,
    3, -1, -3, 1, 1, -3, -1, 3,
    -10, 3, -3, 0, 0, -3, 3, -10
]
full_p = 30
test_pos_time = 0


class GameState(object):
    def __init__(self, chessboard, color):
        self.chessboard = chessboard
        self.color = color
        self.step_number = len(np.where(chessboard)[0])

    def __eq__(self, other):
        if self.get_color != other.get_color:
            return False
        for i in range(64):
            if self.get_chessboard[i] != other.get_chessboard[i]:
                return False
        return True

    def test_position(self, pos):
        for i in range(8):
            x = pos // 8
            y = pos % 8
            p = pos
            while True:
                x += dx[i]
                if x < 0 or x > 7:
                    break
                y += dy[i]
                if y < 0 or y > 7:
                    break
                p += dp[i]
                if self.chessboard[p] == self.color:
                    if p != pos + dp[i]:
                        return True
                    break
                if self.chessboard[p] == COLOR_NONE:
                    break
        return False

    def put_position(self, pos):
        for i in range(8):
            x = pos // 8
            y = pos % 8
            p = pos
            while True:
                x += dx[i]
                if x < 0 or x > 7:
                    break
                y += dy[i]
                if y < 0 or y > 7:
                    break
                p += dp[i]
                if self.chessboard[p] == self.color:
                    while p != pos:
                        p -= dp[i]
                        self.chessboard[p] = self.color
                    break
                if self.chessboard[p] == COLOR_NONE:
                    break
        self.color = -self.color
        self.step_number += 1
        if self.no_step:
            self.color = -self.color
            if self.no_step:
                self.color = COLOR_NONE
        return

    @property
    def get_candidate_list(self):
        candidate_list = []
        for i in range(64):
            if self.chessboard[i] == COLOR_NONE:
                if self.test_position(i):
                    candidate_list.append(i)
        return candidate_list

    @property
    def get_step_number(self):
        return self.step_number

    @property
    def get_chessboard(self):
        return self.chessboard

    @property
    def get_color(self):
        return self.color

    @property
    def is_over(self):
        return self.color == COLOR_NONE

    @property
    def get_winner(self):
        return -np.sign(np.sum(self.chessboard))

    @property
    def no_step(self):
        for i in range(64):
            if self.chessboard[i] == COLOR_NONE:
                if self.test_position(i):
                    return False
        return True

    def random_put(self):
        random_seq = [i for i in range(64)]
        np.random.shuffle(random_seq)
        for i in range(64):
            j = random_seq[i]
            if self.chessboard[j] == COLOR_NONE:
                if self.test_position(j):
                    self.put_position(j)
                    return


class SearchNode(object):
    def __init__(self, game_state, parent, last_pos):
        self.game_state = game_state
        self.parent = parent
        self.last_pos = last_pos
        self.winner_count = 0
        self.result_count = 0
        self.log_result_count = 0
        if self.parent is not None:
            self.parent_color = self.parent.parent_color
        else:
            self.parent_color = self.game_state.get_color
        self.children = []
        self.untried_pos = game_state.get_candidate_list

    def back_propagate(self, winner):
        parent = self
        while parent is not None:
            if winner == parent.parent_color:
                parent.winner_count += 1
            if winner == -parent.parent_color:
                parent.winner_count -= 1
            parent.result_count += 1
            parent.log_result_count = np.log(parent.result_count)
            parent = parent.parent

    @property
    def is_leaf(self):
        return self.game_state.is_over

    @property
    def n(self):
        return self.result_count

    @property
    def logn(self):
        return self.log_result_count

    @property
    def q(self):
        return self.winner_count

    def child_score(self, c, d):
        return [
            ((child.q / child.n) + c * np.sqrt((2 * self.logn / child.n)))
            * ((1 - d) + d * (1 + board_worth[child.last_pos] / full_p))
            for child in self.children
        ]

    def best_child(self, c=1.414, d=0):
        return self.children[np.argmax(self.child_score(c=c, d=d))]

    def rollout(self):
        game_state_now = copy.deepcopy(self.game_state)
        while not game_state_now.is_over:
            game_state_now.random_put()
        return game_state_now.get_winner

    @property
    def is_fully_expanded(self):
        return len(self.untried_pos) == 0

    def expand(self):
        pos = self.untried_pos[-1]
        self.untried_pos = np.delete(self.untried_pos, -1)
        next_state = copy.deepcopy(self.game_state)
        next_state.put_position(pos)
        child_node = SearchNode(game_state=next_state, parent=self, last_pos=pos)
        self.children.append(child_node)
        return child_node

    def find(self, game_state):
        if game_state == self.game_state:
            return self
        for child in self.children:
            result = child.find(game_state)
            if result is not None:
                return result
        return None

    def __str__(self):
        return "last pos = %s, q = %d, n = %d" % (self.last_pos, self.q, self.n)


# don't change the class name
class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        # self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out * 0.95
        self.candidate_list = None
        self.root = None

    def go(self, chessboard):
        end_time = time.time() + self.time_out
        game_state = GameState(np.array(chessboard.flatten(), dtype=np.int8).tolist(), self.color)
        self.candidate_list = game_state.get_candidate_list
        if len(self.candidate_list) == 0:
            return self.candidate_list
        if game_state.get_step_number >= 24:
            self.candidate_list = [(i // 8, i % 8) for i in self.candidate_list]
            if self.root is not None:
                self.root = self.root.find(game_state=game_state)
            if self.root is None:
                self.root = SearchNode(game_state=game_state, parent=None, last_pos=None)
            test_cnt = 0
            while time.time() <= end_time:
                v = self.begin_node()
                result = v.rollout()
                v.back_propagate(result)
                test_cnt += 1

            if local_test:
                print("test_cnt = %d(%d)" % (test_cnt, self.root.result_count))
                score = self.root.child_score(c=0, d=1)
                for i in range(len(self.root.children)):
                    print("%s, point = %f" % (self.root.children[i], score[i]))

            best_pos = self.root.best_child(c=0, d=1).last_pos
        else:
            best_pos = -1
            best_score = -Infinity
            reg_list = self.candidate_list
            self.candidate_list = [(i // 8, i % 8) for i in self.candidate_list]
            for pos in reg_list:
                next_state = copy.deepcopy(game_state)
                next_state.put_position(pos)
                score = board_worth[pos] - len(next_state.get_candidate_list)
                if score > best_score:
                    best_score = score
                    best_pos = pos
                if local_test:
                    print("pos = %d, score = %d - %d = %d" % (pos, board_worth[pos], len(next_state.get_candidate_list), score))

        self.candidate_list.append((best_pos // 8, best_pos % 8))
        return self.candidate_list

    def begin_node(self):
        node_now = self.root
        while not node_now.is_leaf:
            if not node_now.is_fully_expanded:
                return node_now.expand()
            else:
                node_now = node_now.best_child()
        return node_now


if __name__ == "__main__":
    local_test = 1
    test_time = 5
    start_time = time.time()
    myAI = AI(chessboard_size=8, color=COLOR_WHITE, time_out=test_time)
    myAI2 = AI(chessboard_size=8, color=COLOR_BLACK, time_out=test_time)
    board = np.zeros((8, 8))
    board[3][3] = board[4][4] = COLOR_WHITE
    board[3][4] = board[4][3] = COLOR_BLACK
    game_state = GameState(chessboard=board.flatten(), color=COLOR_BLACK)
    print(board)
    test_mode = 2
    while not game_state.is_over:
        pos = (-1, -1)
        if game_state.color == COLOR_BLACK:
            if test_mode == 1:
                print("Player's turn\n")
                x = y = -1
                while x not in range(8) or y not in range(8):
                    x = int(input())
                    y = int(input())
                pos = (x, y)
                print("Player chose:")
                print(pos)
            else:
                print("AI2's turn\n")
                myAI2.go(chessboard=game_state.get_chessboard)
                print("AI2's candidate list:")
                print(myAI2.candidate_list)
                pos = myAI2.candidate_list[-1]
                print("AI2 chose:")
                print(pos)
        else:
            print("AI's turn\n")
            myAI.go(chessboard=game_state.get_chessboard)
            print("AI's candidate list:")
            print(myAI.candidate_list)
            pos = myAI.candidate_list[-1]
            print("AI chose:")
            print(pos)
        game_state.put_position(pos[0] * 8 + pos[1])
        print("Chessboard now:")
        print_chessboard = copy.deepcopy(game_state.get_chessboard)
        print_chessboard.shape = (8, 8)
        print(print_chessboard)

