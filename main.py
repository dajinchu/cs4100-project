import reversi_gym
import gym
import random
import numpy as np
import torch
import torch.optim as optim
from model import QNetworkFC, QNetwork

import sys

torch.manual_seed(123456)

DISCOUNT_FACTOR = 0.8
EXPLORE_PROB = 0.2
INIT_LEARNING_RATE = 0.2
END_LEARNING_RATE = 0.02
ITERATIONS = 1000
MAX_MOVES = 100

def valid_mask(enables):
    # Make a mask of valid moves
    mask = torch.zeros(64)
    mask.scatter_(0, torch.LongTensor(enables), 1.)
    return mask

def train():
    env = gym.make('Reversi8x8-v0')
    env.reset()
    SKIP_ACTION = env.board_size**2 + 1

    qnn = QNetwork()
    qnn.double()
    optimizer = optim.SGD(qnn.parameters(), lr=INIT_LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    # Given a state, return the evaluation of all actions from it

    def Q(state):
        return qnn(torch.from_numpy(state).double())

    wins, ties = 0, 0
    for i_episode in range(ITERATIONS):
        if i_episode % (ITERATIONS/10) == 0:
            step = i_episode / ITERATIONS
            new_lr = INIT_LEARNING_RATE - step * (INIT_LEARNING_RATE-END_LEARNING_RATE)
            print("LEARNING RATE", new_lr)
            optimizer.param_groups[0]['lr'] = new_lr
        previous_eval = None
        previous_action = None
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
                loss.backward()
                # print(reward, max_q, "Qnew", Qnew.detach().numpy(), "LOSS", loss.item())
                optimizer.step()

                # if done:
                #     print("DONE")
                #     Qnew = torch.tensor((reward+1)/2)
                #     new_eval = previous_eval.detach()
                #     new_eval[previous_action] = Qnew
                #     # print(Qnew, Qprev, previous_eval)
                #     optimizer.zero_grad()   # zero the gradient buffers
                #     loss = criterion(previous_eval, new_eval)
                #     loss.backward()
                #     optimizer.step()

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
