import reversi_gym
import gym
import random
import numpy as np

DISCOUNT_FACTOR = 0.8
EXPLORE_PROB = 0.2
LEARNING_RATE = 0.01
ITERATIONS = 100
MAX_MOVES = 100

if __name__ == "__main__":
    print(reversi_gym.__file__)
    env = gym.make('Reversi8x8-v0')
    env.reset()

    # Given a state and action, return the evaluation of it using a NN
    def Q(state, action):
      # TODO

    for i_episode in range(ITERATIONS):
        observation = env.reset()
        for t in range(MAX_MOVES):
            enables = env.possible_actions
            # if nothing to do ,select pass
            if len(enables) == 0:
                action = env.board_size**2 + 1
            # random select (update learning method here)
            else:
                should_explore = random.uniform(0,1) < EXPLORE_PROB
                if should_explore:
                    action = random.choice(enables)
                else:
                    action = enables.argmax(lambda a: Q(observation,a))
            observation, reward, done, info = env.step(action)
            # Adjust the q value
            Qnew = reward + DISCOUNT_FACTOR * max([Q(observation, a) for a in env.possible_actions])
            # Adjust NN to set Q(observation, a) = Qnew

            # env.render()
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                black_score = len(np.where(env.state[0,:,:]==1)[0])
                print(black_score)
                break
