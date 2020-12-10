"""
Game of Reversi.

Original souce code from https://github.com/pigooosuke/gym_reversi
"""

from six import StringIO
import sys
import gym
from gym import spaces
import numpy as np
from gym import error
from gym.utils import seeding
from . import rust_reversi

def make_random_policy(np_random):
    def random_policy(state, player_color):
        possible_places = ReversiEnv.get_possible_actions(state, player_color)
        # No places left
        if len(possible_places) == 0:
            print("Warning: no possible_places left, not sure what to do!")
            return 0
            # return d**2 + 1
        a = np_random.randint(len(possible_places))
        return possible_places[a]
    return random_policy

class ReversiEnv(gym.Env):
    """
    Reversi environment. Play against a fixed opponent.
    """
    BLACK = 0
    WHITE = 1
    metadata = {"render.modes": ["ansi","human"]}

    def __init__(self, player_color, opponent, observation_type, illegal_place_mode, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent policy
            observation_type: State encoding
            illegal_place_mode: What to do when the agent makes an illegal place. Choices: 'raise' or 'lose'
            board_size: size of the Reversi board
        """
        assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size

        colormap = {
            'black': ReversiEnv.BLACK,
            'white': ReversiEnv.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error("player_color must be 'black' or 'white', not {}".format(player_color))

        self.opponent = opponent

        assert observation_type in ['numpy3c']
        self.observation_type = observation_type

        assert illegal_place_mode in ['lose', 'raise']
        self.illegal_place_mode = illegal_place_mode

        if self.observation_type != 'numpy3c':
            raise error.Error('Unsupported observation type: {}'.format(self.observation_type))

        # One action for each board position and resign and pass
        self.action_space = spaces.Discrete(self.board_size ** 2 + 2)
        observation = self.reset()
        self.observation_space = spaces.Box(np.zeros(observation.shape), np.ones(observation.shape))

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = make_random_policy(self.np_random)
            elif self.opponent == 'minimax':
                self.opponent_policy = minimax
            else:
                raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def reset(self):
        # init board setting
        self.state = np.zeros((3, self.board_size, self.board_size))
        centerL = int(self.board_size/2-1)
        centerR = int(self.board_size/2)
        self.state[2, :, :] = 1.0
        self.state[2, (centerL):(centerR+1), (centerL):(centerR+1)] = 0
        self.state[0, centerR, centerL] = 1
        self.state[0, centerL, centerR] = 1
        self.state[1, centerL, centerL] = 1
        self.state[1, centerR, centerR] = 1
        self.to_play = ReversiEnv.BLACK
        self.possible_actions = ReversiEnv.get_possible_actions(self.state, self.to_play)
        self.done = False

        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play:
            a = self.opponent_policy(self.state)
            ReversiEnv.make_place(self.state, a, ReversiEnv.BLACK)
            self.to_play = ReversiEnv.WHITE
        return self.state

    def step(self, action):
        assert self.to_play == self.player_color
        # If already terminal, then don't do anything
        if self.done:
            return self.state, 0., True, {'state': self.state}
        if ReversiEnv.pass_place(self.board_size, action):
            pass
        elif ReversiEnv.resign_place(self.board_size, action):
            return self.state, -1, True, {'state': self.state}
        elif not ReversiEnv.valid_place(self.state, action, self.player_color):
            if self.illegal_place_mode == 'raise':
                self.render()
                raise error.Error('illegal move placing on tile {}'.format(action))
            elif self.illegal_place_mode == 'lose':
                # Automatic loss on illegal place
                self.done = True
                return self.state, -1., True, {'state': self.state}
            else:
                raise error.Error('Unsupported illegal place action: {}'.format(self.illegal_place_mode))
        else:
            ReversiEnv.make_place(self.state, action, self.player_color)

        # Opponent play
        a = self.opponent_policy(self.state, 1 - self.player_color)

        # Making place if there are places left
        if a is not None:
            if ReversiEnv.pass_place(self.board_size, a):
                pass
            elif ReversiEnv.resign_place(self.board_size, a):
                return self.state, 1, True, {'state': self.state}
            elif not ReversiEnv.valid_place(self.state, a, 1 - self.player_color):
                if self.illegal_place_mode == 'raise':
                    raise error.Error("ohno")
                elif self.illegal_place_mode == 'lose':
                    # Automatic loss on illegal place
                    self.done = True
                    return self.state, 1., True, {'state': self.state}
                else:
                    raise error.Error('Unsupported illegal place action: {}'.format(self.illegal_place_mode))
            else:
                ReversiEnv.make_place(self.state, a, 1 - self.player_color)


        self.possible_actions = ReversiEnv.get_possible_actions(self.state, self.player_color)
        reward = ReversiEnv.game_finished(self.state)
        if self.player_color == ReversiEnv.WHITE:
            reward = - reward
        self.done = reward != 0
        return self.state, reward, self.done, {'state': self.state}

    # def _reset_opponent(self):
    #     if self.opponent == 'random':
    #         self.opponent_policy = random_policy
    #     else:
    #         raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))

    def render(self, mode='human', close=False):
        if close:
            return
        board = self.state
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        outfile.write(' ' * 7)
        for j in range(board.shape[1]):
            outfile.write(' ' +  str(j + 1) + '  | ')
        outfile.write('\n')
        outfile.write(' ' * 5)
        outfile.write('-' * (board.shape[1] * 6 - 1))
        outfile.write('\n')
        for i in range(board.shape[1]):
            outfile.write(' ' +  str(i + 1) + '  |')
            for j in range(board.shape[1]):
                if board[2, i, j] == 1:
                    outfile.write('  O  ')
                elif board[0, i, j] == 1:
                    outfile.write('  B  ')
                else:
                    outfile.write('  W  ')
                outfile.write('|')
            outfile.write('\n')
            outfile.write(' ' )
            outfile.write('-' * (board.shape[1] * 7 - 1))
            outfile.write('\n')

        if mode != 'human':
            return outfile

    # @staticmethod
    # def pass_place(board_size, action):
    #     return action == board_size ** 2

    @staticmethod
    def resign_place(board_size, action):
        return action == board_size ** 2

    @staticmethod
    def pass_place(board_size, action):
        return action == board_size ** 2 + 1

    @staticmethod
    def to_rust_board(board):
        # convert to 8x8 array of 0,1,2 (0 is empty, player 1, player 2)
        return (board.transpose((1,2,0)).argmax(axis=2)+1)%3

    @staticmethod
    def from_rust_board(board):
        # convert from 8x8 array of 0,1,2 (0 is empty, player 1, player 2) back to 3x8x8 one hot
        a = ((board-1)%3)
        return (np.arange(a.max()+1) == a[...,None]).astype(int).transpose((2,0,1))

    @staticmethod
    def get_possible_actions(board, player_color):
        rs_board = ReversiEnv.to_rust_board(board)
        rs_color = player_color+1
        return rust_reversi.get_possible_actions(rs_color, rs_board)

    @staticmethod
    def valid_reverse_opponent(board, coords, player_color):
        '''
        check whether there is any reversible places
        '''
        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if(dx == 0 and dy == 0):
                    continue
                nx = pos_x + dx
                ny = pos_y + dy
                n = 0
                if (nx not in range(d) or ny not in range(d)):
                    continue
                while(board[opponent_color, nx, ny] == 1):
                    tmp_nx = nx + dx
                    tmp_ny = ny + dy
                    if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                        break
                    n += 1
                    nx += dx
                    ny += dy
                if(n > 0 and board[player_color, nx, ny] == 1):
                    return True
        return False

    @staticmethod
    def valid_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)
        # check whether there is any empty places
        if board[2, coords[0], coords[1]] == 1:
            # check whether there is any reversible places
            if ReversiEnv.valid_reverse_opponent(board, coords, player_color):
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def make_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)

        pos_x = coords[0]
        pos_y = coords[1]

        #TODO: Have Rust do the mutation?
        rust_board = rust_reversi.place_tile(pos_x, pos_y, player_color+1, ReversiEnv.to_rust_board(board))
        new_board = ReversiEnv.from_rust_board(np.array(rust_board))
        print(new_board, board, coords)
        board[new_board>-9] = new_board[new_board>-9]
        return board


    @staticmethod
    def coordinate_to_action(board, coords):
        return coords[0] * board.shape[-1] + coords[1]

    @staticmethod
    def action_to_coordinate(board, action):
        return action // board.shape[-1], action % board.shape[-1]

    @staticmethod
    def game_finished(board):
        # Returns 1 if player 1 wins, -1 if player 2 wins and 0 otherwise
        d = board.shape[-1]

        player_score_x, player_score_y = np.where(board[0, :, :] == 1)
        player_score = len(player_score_x)
        opponent_score_x, opponent_score_y = np.where(board[1, :, :] == 1)
        opponent_score = len(opponent_score_x)
        if player_score == 0:
            return -1
        elif opponent_score == 0:
            return 1
        else:
            free_x, free_y = np.where(board[2, :, :] == 1)
            if free_x.size == 0:
                if player_score > (d**2)/2:
                    return 1
                elif player_score == (d**2)/2:
                    return 1
                else:
                    return -1
            else:
                return 0
        return 0


WHITE = 1
NOBODY = 0
BLACK = -1

TIE = 2

WIN_VAL = 100


def minimax(state, player_color):
    moves = ReversiEnv.get_possible_actions(state, player_color)
    if moves == [state.size/3 + 1]:
        return moves[0]
    best_val = float("-inf")
    best_move = None
    for move in moves:
        new_state = ReversiEnv.make_place(
            np.copy(state), move, player_color)
        move_val = minimax_value(
            new_state, 1-player_color, 3, float("-inf"), float("inf"))
        if move_val > best_val:
            best_move = move
            best_val = move_val
    return best_move


def minimax_value(state, player_color, search_depth, alpha, beta):
  # board, white_turn, ):
    """Return the value of the board, up to the maximum search depth.

    Assumes white is MAX and black is MIN (even if black uses this function).

    Args:
        board (numpy 2D int array) - The othello board
        white_turn (bool) - True iff white would get to play next on the given board
        search_depth (int) - the search depth remaining, decremented for recursive calls
        alpha (int or float) - Lower bound on the value:  MAX ancestor forbids lower results
        beta (int or float) - Upper bound on the value:  MIN ancestor forbids larger results
    """
    winner = ReversiEnv.game_finished(state)
    if winner == NOBODY:
        return 0
    elif winner == player_color:
        return WIN_VAL
    else:
        return -WIN_VAL

    if search_depth == 0:
        return len(np.where(state[player_color, :, :] == 1))

    children = [ReversiEnv.make_place(state, move, player_color)
                for move in ReversiEnv.get_possible_actions(state, player_color)]
    if not children:
        return minimax_value(state, 1 - player_color, search_depth, alpha, beta)
    if player_color == WHITE:
        max_val = float('-inf')
        for child in children:
            val = minimax_value(child, 1 - player_color,
                                search_depth - 1, alpha, beta)
            max_val = max(max_val, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return max_val
    else:
        min_val = float('inf')
        for child in children:
            val = minimax_value(child, 1 - player_color,
                                search_depth - 1, alpha, beta)
            min_val = min(min_val, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_val
