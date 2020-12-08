from reversi_gym.envs.reversi import ReversiEnv
from model import QNetwork, QNetworkFC
import gym
import random
import torch
from torch.utils.tensorboard import SummaryWriter

import sys

torch.manual_seed(123456)

def generate_states_at_depth(env, d):
    """
    Generator that yields all possible states of Othello at depth d
    """
    ReversiEnv.make_place(env.state, a, ReversiEnv.BLACK)

def test(qnn, name, writer=None):
    env = gym.make('Reversi8x8-v0')
    wins, ties = 0, 0
    for i in range(1000):
        init_obs = env.reset()
        r = test_1_game(env, init_obs, qnn) 
        if r == 1:
          wins += 1
        if r == None:
          ties += 1
        print("{}-{}-{} record. Win ratio: {}".format(wins, i+1-wins-ties, ties, wins/(i+1-ties)))


def test_1_game(env, observation, qnn):
    """
    Play the game with given environment and initial observation
    Returns final reward (who won)
    """
    SKIP_ACTION = env.board_size**2 + 1
    def Q(state):
        return qnn(torch.from_numpy(state).double())
    for t in range(1000):
        # env.render()
        enables = env.possible_actions
        # if nothing to do ,select pass
        if enables == [SKIP_ACTION]:
            action = SKIP_ACTION
            observation, reward, done, info = env.step(action)
        else:
            evaluation = Q(observation)

            # Make a mask of valid moves
            mask = torch.zeros(64)
            mask.scatter_(0, torch.LongTensor(enables), 1.)

            max_q = torch.max(evaluation * mask)
            if torch.isnan(max_q) or max_q == 0:
                action = random.choice(enables)
                print("rand")
                input()
            else:
                action = torch.argmax(evaluation * mask)
            observation, reward, done, info = env.step(action)

            if done:
                return reward

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("please provide an id for the saved weights")
        exit()
    writer = SummaryWriter()
    qnn = QNetwork()
    name = "model_{}".format(sys.argv[1])
    qnn.load_state_dict(torch.load("./trained/{}.pt".format(name)))
    qnn.eval()
    qnn.double()
    test(qnn, name, writer)
    writer.flush()
    writer.close()
