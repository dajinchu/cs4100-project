from gym.envs.registration import register

register(
    id='Reversi8x8-v0',
    entry_point='reversi_gym.envs:ReversiEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_place_mode': 'raise',
        'board_size': 8,
    }
)
