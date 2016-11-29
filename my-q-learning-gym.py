import math
import gym
import numpy.matlib
import numpy as np
import random
import pandas as pd

# Used for discretelize
NUM_BINS = 4
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
    np.random.seed(0)

    EPISLON = 0.5
    EPISLON_DECAY_RATE = 0.99
    ALPHA = 0.05
    GAMMA = 0.99
    NUM_STATES = 10 ** (env.observation_space.shape[0])
    NUM_ACTIONS = (env.action_space.n)
    MAX_EPISODES = 1000
    MAX_TIME_STEPS = 200

    # Trick: shrink the state space
    lows = np.array([-2.4, -3.0, -0.5, -2.0])
    highs = np.array([2.4, 3.0, 0.5, 2.0])
    bounds = np.hstack((np.hsplit(lows, len(lows)), np.hsplit(highs, len(highs))))
    for bound in bounds:
        (out, bins) = pd.qcut(bound, NUM_BINS, retbins=True)
        STATE_BINS.append(bins[1:-1])

    Q = np.zeros((NUM_STATES, NUM_ACTIONS))
    A = np.matlib.repmat(np.arange(NUM_ACTIONS), NUM_STATES, 1)

    for i_episode in xrange(MAX_EPISODES):
        state = env.reset()
        state = discretelize(state)
        for t in xrange(MAX_TIME_STEPS):
            action = epislon_policy(Q, A, state, EPISLON)

            next_state, reward, done, info = env.step(action)
            next_state = discretelize(next_state)

            if done:
                print "Episode {} finished after {} timesteps ( {}, {} )".format(i_episode, t+1, ALPHA, EPISLON)
                if t < MAX_TIME_STEPS - 1:
                    reward = -MAX_TIME_STEPS

            # Step 5. Update Q(s, a)
            Q[state][action] = Q[state][action] + ALPHA * (reward + GAMMA * np.max(Q[next_state]) - Q[state][action])

            state = next_state
            if done:
                break
        EPISLON *= EPISLON_DECAY_RATE
    env.monitor.close()
    gym.upload('/tmp/cartpole-experiment-1', api_key='sk_nRXxhE9sTIqcYJ1DOGDew')
