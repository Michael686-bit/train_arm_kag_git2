import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#####################  hyper parameters  ####################

LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 30000
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 300)
        self.fc2 = nn.Linear(300, a_dim)
        self.a_bound = a_bound

    def forward(self, s):
        x = torch.relu(self.fc1(s))
        a = torch.tanh(self.fc2(x)) * self.a_bound
        return a


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim, 300)
        self.fc2 = nn.Linear(a_dim, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, s, a):
        x = torch.relu(self.fc1(s) + self.fc2(a))
        q = self.fc3(x)
        return q


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]

        self.actor_eval = Actor(s_dim, a_dim, self.a_bound).to(device)
        self.actor_target = Actor(s_dim, a_dim, self.a_bound).to(device)
        self.critic_eval = Critic(s_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim, a_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=LR_A)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=LR_C)

        self.loss_func = nn.MSELoss()

        self.soft_update(self.actor_target, self.actor_eval, 1.0)
        self.soft_update(self.critic_target, self.critic_eval, 1.0)

    def choose_action(self, s):
        s = torch.FloatTensor(s).to(device)
        return self.actor_eval(s).cpu().data.numpy()

    def learn(self):
        if self.pointer < BATCH_SIZE:
            return

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim]).to(device)
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim]).to(device)
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim]).to(device)
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:]).to(device)

        a = self.actor_eval(bs)
        q = self.critic_eval(bs, ba)  # a->ba

        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)

        q_target = br + GAMMA * q_

        td_error = self.loss_func(q, q_target)
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

        a_loss = -self.critic_eval(bs, self.actor_eval(bs)).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor_target, self.actor_eval, TAU)
        self.soft_update(self.critic_target, self.critic_eval, TAU)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:  # indicator for learning
            self.memory_full = True

    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

    def save(self):
        # 指定文件夹路径
        import os
        save_path = './model_save'

        # 确保文件夹存在
        os.makedirs(save_path, exist_ok=True)

        from datetime import datetime

        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'params_{current_time}.pth'  # 文件名
        file_path = os.path.join(save_path, file_name)  # 完整文件路径

        torch.save({
            'actor_eval': self.actor_eval.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_eval': self.critic_eval.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, file_path)

        print(f"save as {file_name}")
        return current_time

    def restore(self):

        import os
        save_path = './model_save'

        # latest_file_name = os.listdir(save_path)[-1]  # 获取最后一个文件名
        # file_path = os.path.join(save_path, latest_file_name)  # 使用传入的文件名

        latest_file_name = "params_20240808_094051.pth"
        file_path = os.path.join(save_path, latest_file_name)  # 使用传入的文件名  params_20240807_194210


        checkpoint = torch.load(file_path)
        self.actor_eval.load_state_dict(checkpoint['actor_eval'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_eval.load_state_dict(checkpoint['critic_eval'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

    # Example usage:
    # ddpg = DDPG(a_dim, s_dim, a_bound)
    # ddpg.save()
    # ddpg.restore('learning_result/model_20240723_225122.pth')

# Example usage:
# ddpg = DDPG(a_dim, s_dim, a_bound)
# while training:
#     ddpg.store_transition(s, a, r, s_)
#     ddpg.learn()
#     action = ddpg.choose_action(s)
