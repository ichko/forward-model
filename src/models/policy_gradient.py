import src.utils.torch as tu

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyGradient(tu.BaseModule):
    def __init__(
        self,
        obs_size,
        num_actions,
    ):
        super().__init__()
        self.name = 'Policy Gradient'

        self.net = nn.Sequential(
            tu.dense(i=obs_size, o=256),
            tu.dense(i=256, o=256),
            tu.dense(i=256, o=num_actions, a=nn.Softmax(dim=0)),
        )

    def forward(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        action_logits = self.net(obs)
        return action_logits

    def configure_optim(self, lr):
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        # def lr_lambda(it):
        #     return lr / (it // 5000 + 1)

        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     self.optim,
        #     lr_lambda=lr_lambda,
        # )

    def optim_step(self, action_memory, reward_memory):
        action_memory = torch.tensor(action_memory, dtype=torch.float32)

        G = np.zeros_like(reward_memory, dtype=np.float32)
        for s in range(len(reward_memory)):
            discount = 1
            for i in range(s, len(reward_memory)):
                G[s] = reward_memory[i] * discount
                discount *= self.gamma

        mean = G.mean()
        std = G.std()
        std = 1 if std == 0 else std
        G = (G - mean) / std

        loss = torch.sum(-G * action_memory)

        if loss.requires_grad:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()

        return loss, {}


class Agent:
    def __init__(self, gamma, obs_size, num_actions):
        self.policy = PolicyGradient(obs_size, num_actions)
        self.gamma = gamma

    def __call__(self, obs):
        probs = self.policy(obs)
        probs = torch.distributions.Categorical(probs)

        action = probs.sample()
        log_probs = probs.log_prob(action)

        return action.item(), log_probs

    def play_episode(self, env):
        obs = env.reset()
        done = False
        score = 0
        action_memory = []
        reward_memory = []

        while not done:
            action, log_prob = self(obs)
            action_memory.append(log_prob)

            obs, reward, done, _info = env.step(action)
            reward_memory.append(reward)
            score += reward

        return score, action_memory, reward_memory

    def optim_step(self, env):
        score, action_memory, reward_memory = self.play_episode(env)
        loss, _ = self.policy.optim_step(action_memory, reward_memory)

        return loss, {'score': score}


def make_model(obs_size, num_actions):
    return PolicyGradient(
        obs_size=obs_size,
        num_actions=num_actions,
    )


def sanity_check():
    num_precondition_frames = 1
    frame_size = (16, 16)
    num_actions = 3
    bs = 10

    model = make_model(
        num_precondition_frames=num_precondition_frames,
        frame_size=frame_size,
        num_actions=num_actions,
    ).to('cuda')

    print(f'RNN NUM PARAMS {model.count_parameters():08,}')
    print(
        f'PRECONDITION FEATURE MAP {model.precondition_out} [{model.flat_precondition_size}]'
    )

    precondition_frames = torch.rand(
        bs,
        num_precondition_frames,
        3,
        *frame_size,
    )
    actions = torch.randint(0, num_actions, size=(bs, ))
    out_frames = model([precondition_frames, actions]).detach().cpu()

    print(f'OUT FRAMES SHAPE {out_frames.shape}')

    model.configure_optim(lr=0.001)
    loss, _info = model.optim_step(
        [[precondition_frames, actions], out_frames], )

    print(f'OPTIM STEP LOSS {loss.item()}')


if __name__ == '__main__':
    sanity_check()
