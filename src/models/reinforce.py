import sys

import src.utils.torch as tu

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

SHOULD_RENDER = '--render' in sys.argv


# SRC - <https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/blob/master/Reinforce/reinforce_torch.py>
class Policy(tu.BaseModule):
    def __init__(self, obs_size, num_actions):
        super().__init__()
        self.name = 'Policy Network (Reinforce)'

        self.net = nn.Sequential(
            tu.conv_encoder(depth=4, sizes=(3, 32, 64, 128, 512)),
            tu.reshape(-1, 512),
            nn.ReLU(inplace=True),
            tu.dense(i=512, o=num_actions, a=nn.Softmax(dim=1)),
        )

    def forward(self, obs):
        obs = T.Tensor(obs).to(self.device)
        return self.net(obs)

    def configure_optim(self, lr):
        self.optim = T.optim.Adam(self.parameters(), lr=lr)

    def optim_step(self, gamma, action_memory, reward_memory):
        G = np.zeros_like(reward_memory, dtype=np.float64)

        for k in range(len(reward_memory)):
            discount = 1
            for i in range(k, len(reward_memory)):
                G[k] += reward_memory[i] * discount
                discount *= gamma
        G = T.tensor(G, dtype=T.float).to(self.device)

        # action_memory = T.tensor(action_memory, dtype=T.float).to(self.device)

        mean = G.mean()
        std = G.std()
        std = 1 if std == 0 else std
        G = (G - mean) / std

        # loss = T.sum(-G * action_memory)

        self.optim.zero_grad()
        loss = 0
        for g, logprob in zip(G, action_memory):
            loss += -g * logprob

        loss.backward()
        self.optim.step()

        return loss, {}


class Agent(tu.BaseModule):
    def __init__(self, gamma, obs_size, num_actions):
        super().__init__()

        self.name = 'AGENT Policy Network (Reinforce)'
        self.policy = Policy(obs_size, num_actions)
        self.gamma = gamma

    def __call__(self, obs):
        probs = self.policy([obs])
        distrib = T.distributions.Categorical(probs)

        action = distrib.sample()
        log_probs = distrib.log_prob(action)

        return action.item(), log_probs

    def play_episode(self, env, max_len):
        obs = env.reset()
        done = False
        score = 0
        action_memory = []
        reward_memory = []
        step = 0

        while not done and step <= max_len:
            step += 1

            action, log_prob = self(obs)
            action_memory.append(log_prob)

            if SHOULD_RENDER:
                env.render()

            obs, reward, done, _info = env.step(action)
            reward_memory.append(reward)
            score += reward

        return score, action_memory, reward_memory

    def optim_step(self, env, max_len):
        score, action_memory, reward_memory = self.play_episode(env, max_len)
        loss, _ = self.policy.optim_step(
            self.gamma,
            action_memory,
            reward_memory,
        )

        return loss.item(), {'score': score}


def make_model(obs_size, num_actions):
    return Agent(
        obs_size=obs_size,
        num_actions=num_actions,
        gamma=0.99,
    )


def sanity_check():
    from src.utils import make_preprocessed_env
    import sneks

    # env = gym.make('LunarLander-v2')
    env = make_preprocessed_env('snek-rgb-16-v1')

    obs_size = 8
    num_actions = 4

    model = make_model(
        obs_size=obs_size,
        num_actions=num_actions,
    ).to('cuda')
    model.policy.configure_optim(lr=0.001)
    model.make_persisted(f'./.models/{model.name}.hdf5')

    print(f'NUM PARAMS {model.count_parameters():08,}')

    try:
        model.preload_weights()
        print('>> weights preloaded')
    except Exception as _e:
        print('>> could not preload')

    n_episodes = 20000
    for i in range(n_episodes):
        loss, info = model.optim_step(env, 1024)
        score = info['score']
        print(f'[EP {i:05}] - LOSS {loss:.5f} - SCORE {score:.2f}')
        model.persist()


if __name__ == '__main__':
    sanity_check()
