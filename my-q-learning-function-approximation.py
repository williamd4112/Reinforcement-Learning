
# coding: utf-8

# In[70]:

import numpy as np
import math
import gym
import itertools
import numpy as np
import pandas as pd
import sys
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler


# In[71]:

env = gym.envs.make("MountainCar-v0")
env.monitor.start('/tmp/MountainCar-v0-experiment-1', force=True)
np.random.seed(0)


# In[72]:

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))


# In[73]:

models = []


# In[74]:

def featurelize(state):
    scaled = scaler.transform([state])
    feature = featurizer.transform(scaled)
    return feature[0]


# In[75]:

def predict(state):
    feature = featurelize(state)
    return np.array([m.predict([feature])[0] for m in models])


# In[76]:

def update(state, action, td_error):
    feature = featurelize(state)
    models[action].partial_fit([feature], [td_error])


# In[77]:

def epislon_policy(s, epislon):
    if (1 - epislon) <= np.random.uniform(0, 1):
         return np.random.choice(env.action_space.n, 1)[0]
    else:
        return np.argmax(predict(s))


# In[78]:

MAX_EPISODE = 500
EPISLON = 0.5
EPISLON_DECAY = 0.99
GAMMA = 0.95

for _ in range(env.action_space.n):
    model = SGDRegressor(learning_rate="constant")
    model.partial_fit([featurelize(env.reset())], [0])
    models.append(model)


# In[79]:

for episode in range(MAX_EPISODE + 1):
    state = env.reset()
    EPISLON *= EPISLON_DECAY
    for t in itertools.count():
        #env.render()
        action = epislon_policy(state, EPISLON)
        next_state, reward, done, _ = env.step(action)
        td_error = reward + GAMMA * np.max(predict(next_state))
        update(state, action, td_error)
        
        print("\rStep {} @ Episode {}/{} ({})".format(t, episode + 1, MAX_EPISODE, reward), end="")
        
        if done:
            break
        state = next_state


# In[80]:

env.monitor.close()
gym.upload('/tmp/MountainCar-v0-experiment-1', api_key='sk_nRXxhE9sTIqcYJ1DOGDew')


# In[ ]:




# In[ ]:



