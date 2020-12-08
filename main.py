import reversi_gym
import gym
import random
import numpy as np
import torch
import math
from collections import deque
from statistics import mean
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import QNetworkFC, QNetwork

import sys

torch.manual_seed(123456)

DISCOUNT_FACTOR = 0.8
INIT_EXPLORE_PROB = 0.9
END_EXPLORE_PROB = 0.05
INIT_LEARNING_RATE = 0.2
END_LEARNING_RATE = 0.02
ITERATIONS = 1000
MAX_MOVES = 100

def valid_mask(enables):
    # Make a mask of valid moves
    mask = torch.zeros(64)
    mask.scatter_(0, torch.LongTensor(enables), 1.)
    return mask

# linear interpolation from start to end using step value [0,1]
def interpolate(start,end,step):
    return start + (end-start) * step

# INsert new_sample into moving average of window size
avgs = {}
def moving_avg(key,size,new_sample):
    if key not in avgs:
        avgs[key] = deque([new_sample])
    else:
        avgs[key].appendleft(new_sample)
        if len(avgs[key]) == size:
            avgs[key].pop()
    return mean(avgs[key])
  

def train(name, writer=None):
    env = gym.make('Reversi8x8-v0')
    env.reset()
    SKIP_ACTION = env.board_size**2 + 1

    qnn = QNetworkFC()
    qnn.double()
    optimizer = optim.SGD(qnn.parameters(), lr=INIT_LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    # Given a state, return the evaluation of all actions from it

    def Q(state):
        return qnn(torch.from_numpy(state).double())

    wins, ties = 0, 0
    for i_episode in range(ITERATIONS):
        step = i_episode / ITERATIONS
        # interpolate learning rate
        new_lr = interpolate(INIT_LEARNING_RATE, END_LEARNING_RATE, step)
        optimizer.param_groups[0]['lr'] = new_lr
        #interpolate explore probability
        EXPLORE_PROB = interpolate(INIT_EXPLORE_PROB, END_EXPLORE_PROB, step)
        observation = env.reset()
        losses = []
        for t in range(MAX_MOVES):
            # env.render()
            enables = env.possible_actions
            # if nothing to do ,select pass
            if enables == [SKIP_ACTION]:
                action = SKIP_ACTION
                observation, reward, done, info = env.step(action)
            else:
                evaluation = Q(observation)

                mask = valid_mask(enables)
                max_q = torch.max(evaluation * mask)

                should_explore = random.uniform(0, 1) < EXPLORE_PROB
                if should_explore or torch.isnan(max_q) or max_q == 0:
                    # Random if exploring or if every action has evaluation 0
                    action = random.choice(enables)
                else:
                    action = torch.argmax(evaluation * mask)
                previous_eval = evaluation.clone()
                previous_action = action
                observation, reward, done, info = env.step(action)

                # Adjust NN to set Q(observation, a) = Qnew
                if(done):
                    Qnew = torch.tensor((reward+1)/2) # scale range [-1,1] to [0,1]
                else:
                    max_q = torch.max(Q(observation) * mask)
                    Qnew = DISCOUNT_FACTOR * max_q
                new_eval = previous_eval.detach().clone()
                new_eval[previous_action] = Qnew
                optimizer.zero_grad()   # zero the gradient buffers
                loss = criterion(previous_eval, new_eval)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            if done:
                print("Episode #{}/{} finished after {} timesteps".format(i_episode, ITERATIONS, t+1))
                black_score = len(np.where(env.state[0, :, :] == 1)[0])
                print("score ", black_score)
                if reward == 1:
                    wins += 1
                if reward == None:
                    ties += 1
                if i_episode % 10 ==0:
                    print("{}-{}-{} record. Win ratio: {}".format(wins, 1+i_episode-wins-ties, ties, wins/(1+i_episode-ties)))
                if writer:
                    writer.add_scalar("{}/win_moving_avg/train".format(name), moving_avg("wins", 20, reward), i_episode)
                    writer.add_scalar("{}/avg_loss_episode/train".format(name), mean(losses), i_episode)
                # clear loss for next episode
                losses = []
                break

    print("{}-{}-{} record. Win ratio: {}".format(wins, ITERATIONS-wins-ties, ties, wins/(ITERATIONS-ties)))
    return qnn


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("please provide an id for the saved weights")
        exit()
    print(reversi_gym.__file__)
    writer = SummaryWriter()
    name = "model_{}".format(sys.argv[1])
    model = train(name, writer)
    writer.flush()
    writer.close()
    torch.save(model.state_dict(), "./trained/{}.pt".format(name))
