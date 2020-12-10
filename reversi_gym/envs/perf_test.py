from timeit import timeit
from reversi import ReversiEnv

board = ReversiEnv('black',"random", "numpy3c", "raise", 8).reset()
print(ReversiEnv.make_place(board, 26, 0))
print("make_place simple:\n", timeit(lambda: ReversiEnv.make_place(board, 26, 0), number=10000))
print("get_possible simple:\n", timeit(lambda: ReversiEnv.get_possible_actions(board, 0), number=10000))
print("valid_place simple:\n", timeit(lambda: ReversiEnv.valid_place(board, 26, 0), number=10000))

rs_board = ReversiEnv.to_rust_board(board)
print("to_rust_board simple:\n", timeit(lambda: ReversiEnv.to_rust_board(board), number=10000))
print("from_rust_board simple:\n", timeit(lambda: ReversiEnv.from_rust_board(rs_board), number=10000))