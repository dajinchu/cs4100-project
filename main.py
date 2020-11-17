import reversi_gym
import gym
import random
import numpy as np
import torch
import torch.optim as optim
from model import QNetwork 

DISCOUNT_FACTOR = 0.8
EXPLORE_PROB = 0.2
LEARNING_RATE = 0.01
ITERATIONS = 1000
MAX_MOVES = 100

if __name__ == "__main__":
    print(reversi_gym.__file__)
    env = gym.make('Reversi8x8-v0')
    env.reset()

    qnn = QNetwork()
    qnn.double()
    optimizer = optim.SGD(qnn.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    # Given a state, return the evaluation of all actions from it
    def Q(state):
        s_tensor = torch.from_numpy(state).double()
        a = qnn(s_tensor)
        return a

    wins = 0

    for i_episode in range(ITERATIONS):
        Qprev = False
        observation = env.reset()
        for t in range(MAX_MOVES):
            # env.render()
            enables = env.possible_actions
            # if nothing to do ,select pass
            if enables == [env.board_size**2 + 1]:
                action = env.board_size**2 + 1
                observation, reward, done, info = env.step(action)
            else:

                # Adjust the q value
                if Qprev:
                    evaluation = Q(observation)
                    mask = torch.zeros(64)
                    mask.scatter_(0, torch.LongTensor(env.possible_actions), 1.)

                    Qnew = reward + DISCOUNT_FACTOR * torch.max(evaluation * mask)
                    # Adjust NN to set Q(observation, a) = Qnew
                    optimizer.zero_grad()   # zero the gradient buffers
                    loss = criterion(Qprev, Qnew)
                    loss.backward()
                    optimizer.step() 
                    
                    # print('qnn.fc1.grad after backward')
                    # print(qnn.fc1.bias.grad)



                should_explore = random.uniform(0,1) < EXPLORE_PROB
                evaluation = Q(observation) # TODO we run Q twice per loop
                if should_explore:
                    action = random.choice(enables)
                else:
                    # Make a mask of valid moves
                    mask = torch.zeros(64)
                    mask.scatter_(0, torch.LongTensor(enables), 1.)
                    # Random if everything is 0
                    if torch.max(evaluation * mask) == 0:
                        action = random.choice(enables)
                    else:
                        action = torch.argmax(evaluation * mask)
                Qprev = evaluation[action]
                observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                black_score = len(np.where(env.state[0,:,:]==1)[0])
                print("score ", black_score)
                if reward > 0:
                    wins += 1
                if reward == 0:
                    wins += .5
                print("win rate ", wins, "/", i_episode+1, " = ", wins/(i_episode+1))
                # running_avg = ((running_avg * i_episode) + black_score) / (i_episode + 1)
                # print("running avg ", running_avg)
                break
