import math
import gym
import numpy.matlib
import numpy as np
import random
import pandas as pd

# Used for discretelize
NUM_BINS = 10
STATE_BINS = []

def discretelize(s):
    ds = np.array(s, dtype=int)
    for i in xrange(len(s)):
        ds[i] = (np.digitize(s[i], STATE_BINS[i]))
    return int("".join(map(lambda feature: str(int(feature)), ds)))

def epislon_policy(Q, A, s, epislon):
    if (1 - epislon) <= np.random.uniform(0, 1):
         return np.random.choice(A[s])
    else:
        return np.argmax(Q[s])


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    env.monitor.start('/tmp/cartpole-experiment-1', force=True)

    # Step 0. Setup constants
    EPISLON = 0.25
    EPISLON_DECAY_RATE = 0.99
    ALPHA = 0.25
    ALPHA_DECAY_RATE = 0.99
    GAMMA = 1.0
    NUM_STATES = 10 ** (env.observation_space.shape[0])
    NUM_ACTIONS = (env.action_space.n)
    MAX_EPISODES = 300
    MAX_TIME_STEPS = 1000

    # Step 1. preparaed discretelization params
    lows = env.observation_space.low
    highs = env.observation_space.high
    bounds = np.hstack((np.hsplit(lows, len(lows)), np.hsplit(highs, len(highs))))
    for bound in bounds:
        (out, bins) = pd.qcut(bound, NUM_BINS, retbins=True)
        STATE_BINS.append(bins[1:-1])

    Q = np.random.uniform(low=0, high=1, size=(NUM_STATES, NUM_ACTIONS))
    A = np.matlib.repmat(np.arange(NUM_ACTIONS), NUM_STATES, 1)

    for i_episode in xrange(MAX_EPISODES):
        state = env.reset()
        state = discretelize(state)
        for t in xrange(MAX_TIME_STEPS):
            env.render()

            # Step 4. Under s, take action from A(s)
            action = epislon_policy(Q, A, state, EPISLON)

            next_state, reward, done, info = env.step(action)
            next_state = discretelize(next_state)

            if done:
                print "Episode {} finished after {} timesteps ( {}, {} )".format(i_episode, t+1, ALPHA, EPISLON)
                break

            # Step 5. Update Q(s, a)
            Q[state][action] = Q[state][action] + ALPHA * (reward + GAMMA * np.max(Q[next_state]) - Q[state][action])

            EPISLON *= EPISLON_DECAY_RATE
            ALPHA *= ALPHA_DECAY_RATE
            state = next_state
    env.monitor.close()
    gym.upload('/tmp/cartpole-experiment-1', api_key='sk_nRXxhE9sTIqcYJ1DOGDew')
