from __future__ import annotations
from collections import defaultdict
import random
from typing import List, Tuple

State = Tuple[str, str, str]

class Chesschess:
    def __init__(self, state: State = None, agentX: Agent = None, agentY: Agent = None, headless = True):
        self.agentX = agentX
        self.agentY = agentY
        self.history = []
        self.headless = headless
        if not state:
            self.board = [[' ' for _ in range(3)] for _ in range(3)]
            self.players = ['X', 'O']
            self.current_player = 0
            self.sizes = {'L': 2, 'M': 3, 'S': 3}
            self.size_order = {'S': 0, 'M': 1, 'L': 2}
            self.pieces = {player: dict(self.sizes) for player in self.players}
        else:
            self.board = [[],[],[]]
            board_str, my_piece_count, opponent_piece_count = state
            for x in range(3):
                for y in range(3):
                    cell_str = board_str[y * 3 + x]
                    if cell_str.isupper():
                        self.board[y].append('X' + cell_str)
                    elif cell_str != ' ':
                        self.board[y].append('O' + cell_str.upper())
                    else:
                        self.board[y].append(' ')
            self.players = ['X', 'O']
            self.current_player = 0
            self.sizes = {'L': 2, 'M': 3, 'S': 3}
            self.size_order = {'S': 0, 'M': 1, 'L': 2}
            # Convert 233 to {L: 2, M: 3, S: 3}
            my_piece_count = {size: int(my_piece_count[i]) for i, size in enumerate(self.sizes)}
            opponent_piece_count = {size: int(opponent_piece_count[i]) for i, size in enumerate(self.sizes)}
            self.pieces = {self.players[0]: my_piece_count, self.players[1]: opponent_piece_count}

    # Get the current state of the board, in a hashable format
    def get_state(self):
        board = ""
        for row in self.board:
            for cell in row:
                content = cell
                if content != ' ':
                    content = content[1] if content[0] == self.players[self.current_player] else content[1].lower()
                board += content
        my_piece_count = ''.join(str(self.pieces[self.players[self.current_player]][size]) for size in self.sizes)
        opponent_piece_count = ''.join(str(self.pieces[self.players[1 - self.current_player]][size]) for size in self.sizes)

        return (board, my_piece_count, opponent_piece_count)

    def print_board(self):
        for row in self.board:
            print('|' + '|'.join(row) + '|')
        # print()

    def check_win(self, player):
        b = self.board
        lines = (
            b[0], b[1], b[2],
            [b[0][0], b[1][0], b[2][0]],
            [b[0][1], b[1][1], b[2][1]],
            [b[0][2], b[1][2], b[2][2]],
            [b[0][0], b[1][1], b[2][2]],
            [b[0][2], b[1][1], b[2][0]],
        )
        # Check if any line has all cells occupied by the current player's pieces
        return any(all(cell != ' ' and cell[0] == player for cell in line) for line in lines)

    def place_piece(self, x, y, size, player):
        # Check that the there is still a piece of the given size
        if self.pieces[player][size] == 0:
            return False

        if self.board[y][x] == ' ' or self.size_order[self.board[y][x][1]] < self.size_order[size]:
            self.board[y][x] = player + size
            self.pieces[player][size] -= 1  # Decrement the count of available pieces
            return True
        return False

    def switch_player(self):
        self.current_player = 1 - self.current_player

    def get_valid_actions(self):
        valid_actions = []
        for x in range(3):
            for y in range(3):
                for size in self.sizes:
                    dummy_game = Chesschess(self.get_state())
                    if dummy_game.place_piece(x, y, size, 'X'):
                        valid_actions.append((x, y, size))
        return valid_actions

    def play(self):
        while True:
            if not self.headless:
                print('state:', self.get_state())
                self.print_board()
            # print(self.get_valid_actions())
            # x, y, size = self.get_move()

            state = self.get_state()
            valid_actions = self.get_valid_actions()

            if self.current_player == 0:
                # If the agent is a QAgent, update the Q-table with the previous state and action
                if len(self.history) >= 2 and isinstance(self.agentX, QAgent):
                    self.agentX.update(self.history[-2][0], self.history[-2][1], state, valid_actions, 0)
                action = self.agentX.get_action(state, valid_actions)
            else:
                if len(self.history) >= 2 and isinstance(self.agentY, QAgent):
                    self.agentY.update(self.history[-2][0], self.history[-2][1], state, valid_actions, 0)
                action = self.agentY.get_action(state, valid_actions)

            x, y, size = action

            if not self.headless:
                print(f"Player {self.players[self.current_player]} places a {size} piece at ({x}, {y}).\n\n")
            if self.place_piece(x, y, size, self.players[self.current_player]):
                if self.check_win(self.players[self.current_player]):
                    if not self.headless:
                        self.print_board()
                        print(f"Player {self.players[self.current_player]} wins!")

                    winner = self.players[self.current_player]

                    # If the agent is a QAgent, update the Q-table with the previous state and action
                    if isinstance(self.agentX, QAgent):
                        self.agentX.update(state, action, self.get_state(), ["end"], 1)
                    if isinstance(self.agentY, QAgent):
                        self.switch_player()
                        self.agentY.update(self.history[-1][0], self.history[-1][1], self.get_state(), ["end"], -1)

                    return winner
                self.switch_player()
            else:
                print("Invalid move, that spot is taken by a larger or same size piece.")

            if len(self.get_valid_actions()) == 0:
                if not self.headless:
                    self.print_board()
                    print("The game is a draw!")
                if isinstance(self.agentX, QAgent):
                    self.agentX.update(state, action, self.get_state(), ["end"], 0)
                if isinstance(self.agentY, QAgent):
                    self.switch_player()
                    self.agentY.update(self.history[-1][0], self.history[-1][1], self.get_state(), ["end"], 0)

                return 'draw'
            
            self.history.append((state, action))

class Agent:
    def get_action(self, state, valid_actions):
        pass

class HumanAgent(Agent):
    def get_action(self, state, valid_actions):
        while True:
            try:
                coord, size = input(f"Player, enter your move (coordinate size): ").split()
                x, y = (int(coord) - 1) % 3, (int(coord) - 1) // 3
                if (x, y, size) in valid_actions:
                    return x, y, size
                else:
                    print("Invalid move. Please enter a valid move.")
            except ValueError:
                print("Invalid input. Please enter a valid move.")

class RandomAgent(Agent):
    def get_action(self, state, valid_actions):
        return random.choice(valid_actions)

# A Q-learning agent that learns to play Chesschess
class QAgent(Agent):

    @staticmethod
    def rotate_board_clockwise(board: str):
        return board[6] + board[3] + board[0] + board[7] + board[4] + board[1] + board[8] + board[5] + board[2]
    
    @staticmethod
    def reflect_board_horizontal(board: str):
        return board[6] + board[7] + board[8] + board[3] + board[4] + board[5] + board[0] + board[1] + board[2]
    
    @staticmethod
    def rotate_action_clockwise(action: Tuple[int, int, str]):
        x, y, size = action
        return (2 - y, x, size)
    
    @staticmethod
    def reflect_action_horizontal(action: Tuple[int, int, str]):
        x, y, size = action
        return (2 - x, y, size)
    
    def __init__(self):
        manager = multiprocessing.Manager()
        self.q_table = manager.dict()
        self.lock = manager.Lock()
        # self.q_table = defaultdict(int)
        self.alpha = 0.3
        self.gamma = 0.9
        # self.gamma = 1
        self.epsilon = 0.1
        self.training = True

    # Reduce the state space by matching the rotational, reflectional, and piece size symmetries, return transformed state and action, history of transformation
    def get_q_value(self, state, action):
        board, my_piece_count, opponent_piece_count = state
        if action == "end":
            return 0
        my_piece_repr = ''
        for i in my_piece_count:
            if i == '0':
                my_piece_repr += '0'
            else:
                my_piece_repr += '1'
        opponent_piece_repr = ''
        for i in opponent_piece_repr:
            if i == '0':
                opponent_piece_repr += '0'
            else:
                opponent_piece_repr += '1'

        with self.lock:

            for reflected, num_rot in [
                (False, 0),
                (False, 1),
                (False, 2),
                (False, 3),
                (True, 0),
                (True, 1),
                (True, 2),
                (True, 3),
            ]:
                transformed_board = self.reflect_board_horizontal(board) if reflected else board
                transformed_action = self.reflect_action_horizontal(action) if reflected else action
                for _ in range(num_rot):
                    transformed_board = self.rotate_board_clockwise(transformed_board)
                    transformed_action = self.rotate_action_clockwise(transformed_action)

                transformed_state = (transformed_board, my_piece_count, opponent_piece_count)

                if transformed_state in self.q_table:
                    if transformed_action in self.q_table[transformed_state]:
                        return self.q_table[transformed_state][transformed_action]
                    else:
                        return 0
                    # return (transformed_board, my_piece_count, opponent_piece_count), transformed_action, (reflected, num_rot)

        return 0
            

    # def get_q_value(self, state, action):
    #     with self.lock:
    #         if (state, action) not in self.q_table:
    #             return 0
    #     return self.q_table[(state, action)]
    
    def set_q_value(self, state, action, value):
        
        board, my_piece_count, opponent_piece_count = state
        my_piece_repr = ''
        for i in my_piece_count:
            if i == '0':
                my_piece_repr += '0'
            else:
                my_piece_repr += '1'
        opponent_piece_repr = ''
        for i in opponent_piece_count:
            if i == '0':
                opponent_piece_repr += '0'
            else:
                opponent_piece_repr += '1'

        with self.lock:

            for reflected, num_rot in [
                (False, 0),
                (False, 1),
                (False, 2),
                (False, 3),
                (True, 0),
                (True, 1),
                (True, 2),
                (True, 3),
            ]:
                transformed_board = self.reflect_board_horizontal(board) if reflected else board
                transformed_action = self.reflect_action_horizontal(action) if reflected else action
                for _ in range(num_rot):
                    transformed_board = self.rotate_board_clockwise(transformed_board)
                    transformed_action = self.rotate_action_clockwise(transformed_action)

                transformed_state = (transformed_board, my_piece_count, opponent_piece_count)

                if transformed_state in self.q_table:
                    self.q_table[transformed_state][transformed_action] = value
                    return
                    # return (transformed_board, my_piece_count, opponent_piece_count), transformed_action, (reflected, num_rot)

        self.q_table[(board, my_piece_repr, opponent_piece_repr)] = {action: value}
            

    # def get_valid_actions(self, state):
    #     valid_actions = []
    #     for x in range(3):
    #         for y in range(3):
    #             for size in game.sizes:
    #                 dummy_game = Chesschess(state)
    #                 if dummy_game.place_piece(x, y, size, 'X'):
    #                     valid_actions.append((x, y, size))
    #     return valid_actions

    def get_best_action(self, state, valid_actions):
        
        # valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return None
        return max(valid_actions, key=lambda action: self.get_q_value(state, action))
    
    def get_action(self, state, valid_actions):
        if random.random() < self.epsilon and self.training:
            return random.choice(valid_actions)
        return self.get_best_action(state, valid_actions)
    
    def update(self, state, action, next_state, valid_actions, reward):
        if self.training:
            best_next_action = self.get_best_action(next_state, valid_actions)
            td_target = reward + self.gamma * self.get_q_value(next_state, best_next_action)
            td_delta = td_target - self.get_q_value(state, action)
            new_value = self.get_q_value(state, action) + self.alpha * td_delta
            self.set_q_value(state, action, new_value)

    def get_q_entry_iterator(self):
        with self.lock:
            for state in self.q_table:
                for action in self.q_table[state]:
                    yield state, action, self.q_table[state][action]

    def get_q_table(self):
        with self.lock:
            return dict(self.q_table)
        
    def set_q_table(self, q_table):
        with self.lock:
            self.q_table.update(q_table)

def agentStrToClass(agentStr, defaultQAgent = None):
    if agentStr == "R":
        return RandomAgent()
    elif agentStr == "Q":
        defaultQAgent.training = True
        return defaultQAgent
    elif agentStr == "H":
        return HumanAgent()
    elif agentStr == "QQ":
        defaultQAgent.training = False
        return defaultQAgent

# Time each run of the game
import tqdm
import multiprocessing
from itertools import repeat
import pickle

def run_turn(args):

    agentX_str, agentY_str, QAgentObj = args
    agentX = agentStrToClass(agentX_str, QAgentObj)
    agentY = agentStrToClass(agentY_str, QAgentObj)
    game = Chesschess(agentX = agentX, agentY = agentY)
    outcome = game.play()
    return outcome


if __name__ == "__main__":
    QAgentObj = QAgent()
    while True:
        print("Welcome to Chesschess!")
        print("setup game, specify agent for X, agent for Y, then the number of turns to play")
        print("Example: R R 10")

        command = input()
        if command == "stats":
            # print(QAgentObj.q_table)
            # Summary the Q-table, show number of states and actions with non-zero Q-values
            non_zero = 0
            total = 0
            # for state, action in QAgentObj.q_table:
            #     if QAgentObj.q_table[(state, action)] != 0:
            #         non_zero += 1
            for state in QAgentObj.q_table:
                for action in QAgentObj.q_table[state]:
                    if QAgentObj.q_table[state][action] != 0.0:
                        non_zero += 1
                    total += 1
            print(f"Number of states: {total}")
            print(f"Number of zero Q-values: {total - non_zero}")
            print(f"Number of non-zero Q-values: {non_zero}")
            continue

        if command == "print": #Print the first 10 states and actions in the Q-table
            for i, (state, action, value) in enumerate(QAgentObj.get_q_entry_iterator()):
                if i >= 10:
                    break
                print("State:", state, "Action:", action, "Q-value:", value)
            continue

        if command == "save":
            with open("q_table.pkl", "wb") as f:
                pickle.dump(QAgentObj.get_q_table(), f)
            continue

        if command == "load":
            with open("q_table.pkl", "rb") as f:
                QAgentObj.set_q_table(pickle.load(f))
            continue

        agentX_str, agentY_str, num_turns = command.split()

        if agentX_str == "H" or agentY_str == "H":
            agentX = agentStrToClass(agentX_str, QAgentObj)
            agentY = agentStrToClass(agentY_str, QAgentObj)
            game = Chesschess(agentX = agentX, agentY = agentY, headless=False)
            game.play()
            continue

        X_win = 0
        O_win = 0
        draw = 0

        # with Pool(12) as p:
        #     r = list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))
        #     print(r)

        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:

            # Create a list of arguments for the worker function
            # f = lambda _: run_turn(agentX_str, agentY_str, QAgentObj)

            # Use tqdm to create the progress bar
            for i in tqdm.tqdm(pool.imap_unordered(run_turn, repeat((agentX_str, agentY_str, QAgentObj), int(num_turns))), total=int(num_turns)):
                if i == 'X':
                    X_win += 1
                elif i == 'O':
                    O_win += 1
                else:
                    draw += 1

        print(f"X wins: {X_win}, O wins: {O_win}, Draws: {draw}")