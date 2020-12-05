import reversi_gym.envs.reversi as reversi
import numpy as np

WHITE = 1
NOBODY = 0
BLACK = -1

TIE = 2

WIN_VAL = 100


def minimax(state, player_color):
    moves = reversi.ReversiEnv.get_possible_actions(state, player_color)
    if moves == [state.size/3 + 1]:
        return moves[0]
    best_val = float("-inf")
    best_move = None
    for move in moves:
        new_state = reversi.ReversiEnv.make_place(
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
    winner = reversi.ReversiEnv.game_finished(state)
    if winner == NOBODY:
        return 0
    elif winner == player_color:
        return WIN_VAL
    else:
        return -WIN_VAL

    if search_depth == 0:
        return len(np.where(state[player_color, :, :] == 1))

    children = [reversi.ReversiEnv.make_place(state, move, player_color)
                for move in reversi.ReversiEnv.get_possible_actions(state, player_color)]
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
