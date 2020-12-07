import reversi_gym
import gym
import random
import numpy as np
import torch
import torch.optim as optim
from model import QNetwork

import sys

torch.manual_seed(1234567)

DISCOUNT_FACTOR = 0.08
EXPLORE_PROB = 0.2
LEARNING_RATE = 0.1
ITERATIONS = 1000
MAX_MOVES = 100


def train():
    env = gym.make('Reversi8x8-v0')
    env.reset()
    SKIP_ACTION = env.board_size**2 + 1

    qnn = QNetwork()
    qnn.double()
    optimizer = optim.SGD(qnn.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    # Given a state, return the evaluation of all actions from it

    def Q(state):
        return qnn(torch.from_numpy(state).double())

    wins, ties = 0, 0
    for i_episode in range(ITERATIONS):
        Qprev = False
        observation = env.reset()
        for t in range(MAX_MOVES):
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
                # Adjust the q value
                if Qprev:
                    # Adjust NN to set Q(observation, a) = Qnew
                    Qnew = reward + DISCOUNT_FACTOR * max_q
                    optimizer.zero_grad()   # zero the gradient buffers
                    loss = criterion(Qprev, Qnew)
                    loss.backward()
                    optimizer.step()

                should_explore = random.uniform(0, 1) < EXPLORE_PROB
                if should_explore or torch.isnan(max_q) or max_q == 0:
                    # Random if exploring or if every action has evaluation 0
                    action = random.choice(enables)
                else:
                    action = torch.argmax(evaluation * mask)
                Qprev = evaluation.detach()[action]
                observation, reward, done, info = env.step(action)

            if done:
                print("Episode #{}/{} finished after {} timesteps".format(i_episode, ITERATIONS, t+1))
                black_score = len(np.where(env.state[0, :, :] == 1)[0])
                print("score ", black_score)
                if reward == 1:
                    wins += 1
                if reward == None:
                    ties += 1
                break
    print("{}-{}-{} record. Win ratio: {}".format(wins, ITERATIONS-wins-ties, ties, wins/(ITERATIONS-ties)))
    return qnn


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("please provide an id for the saved weights")
        exit()
    print(reversi_gym.__file__)
    model = train()
    torch.save(model.state_dict(), "./trained/model_{}.pt".format(sys.argv[1]))
