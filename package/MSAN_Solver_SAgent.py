import numpy as np
import time, datetime
import matplotlib.pyplot as plt
import torch
import torchvision
from pytorch_model_summary import summary
from torch import nn
from torchvision import transforms as T
from PIL import Image
from torchstat import stat
from pathlib import Path
from collections import deque
import random, datetime, os, copy
import torch.nn.functional as F
from torch.distributions import Normal, Beta
import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class TTENet(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        """
        N=(W-F+2P)/S+1
        其中N：输出大小
        W：输入大小
        F：卷积核大小
        P：填充值的大小
        S：步长大小
        """
        self.size_linear = (w-4)//3+1
        self.mid_linear = (output_dim+self.size_linear*32)//2
        kernel = 4
        #self.mid_linear = 1000
        # self.online = nn.Sequential(
        #     nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(4, 4), stride=3),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(self.size_linear*32, self.mid_linear),
        #     nn.ReLU(),
        #     nn.Linear(self.mid_linear, output_dim),
        # )
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            #nn.Normo
            nn.Flatten()
        )

        self.target = copy.deepcopy(self.online)
        # Q_target parameters are frozen.
        for p in self.target.parameters():  # 目标Q网络不需要反向传播，因此需要对其所有的参数设定.requires_grad=False
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


class DQN(TTENet):  # DDPG_Actor
    def __init__(self, state_dim, output_dim):
        super(DQN, self).__init__(state_dim, output_dim)
        self.Line = nn.Linear(output_dim, output_dim)

    def forward(self, state, model="online"):
        v = self.online(state)  # 输出到1
        v = self.Line(v)
        if model=="target":
            v = self.target(state)  # 输出到1
            with torch.no_grad():
                v = self.Line(v)
        return v


class DDQN(TTENet):  # DDPG_Actor
    def __init__(self, state_dim, output_dim):
        super(DDQN, self).__init__(state_dim, output_dim)
        self.Line = nn.Linear(output_dim, output_dim)

    def forward(self, state, model):
        if model == "online":
            v = self.online(state)  # 输出到1
            v = self.Line(v)
        if model=="target":
            v = self.target(state)  # 输出到1
            with torch.no_grad():
                v = self.Line(v)
        return v



class DulineDQN(TTENet):  # DDPG_Actor
    def __init__(self, state_dim, output_dim):
        super(DulineDQN, self).__init__(state_dim, output_dim)
        self.Line = nn.Linear(output_dim, output_dim)
        self.V = nn.Linear(output_dim, 1)
        self.targetLine = copy.deepcopy(self.Line)
        self.targetV = copy.deepcopy(self.V)

        # Q_target parameters are frozen.
        for p in self.targetV.parameters():  # 目标Q网络不需要反向传播，因此需要对其所有的参数设定.requires_grad=False
            p.requires_grad = False
        for p in self.targetLine.parameters():  # 目标Q网络不需要反向传播，因此需要对其所有的参数设定.requires_grad=False
            p.requires_grad = False

    def forward(self, state, model):
        if model == "online":
            x = self.online(state)
            A = self.Line(x)
            V = self.V(x)
            return A, V
        elif model == "target":
            x = self.target(state)
            A = self.targetLine(x)
            V = self.targetV(x)
            return A, V


class DPG(TTENet):  # DDPG_Actor
    def __init__(self, state_dim, output_dim):
        super(DPG, self).__init__(state_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state, model="online"):
        v = self.online(state) 
        v = self.softmax(v)
        return v


class DDPG_Actor(TTENet):  # DDPG_Actor
    def __init__(self, state_dim, output_dim):
        super(DDPG_Actor, self).__init__(state_dim, output_dim)
        self.tahn = nn.Tanh()
        self.fc1 = nn.Linear(output_dim, 1)

    def forward(self, state, model="online"):
        # Model
        x = self.online(state)

        x = self.fc1(x)
        return x


class DDPG_Critic(TTENet):  # DDPG_Critic
    def __init__(self, state_dim, output_dim):
        super(DDPG_Critic, self).__init__(state_dim, output_dim)
        self.fc3 = nn.Linear(output_dim+1, 1)

    def forward(self, state, a):
        s = self.online(state)
        #x = torch.cat(s, a))
        x = torch.cat([s, a], dim=1)
        x = self.fc3(x)
        return x

class GaussianActor_musigma(TTENet):
    def __init__(self, state_dim, output_dim, action_dim):
        super(GaussianActor_musigma, self).__init__(state_dim, output_dim)

        self.mu_head = nn.Linear(output_dim, action_dim)
        self.sigma_head = nn.Linear(output_dim, action_dim)

    def forward(self, state, model):
        if model == "online":
            a = torch.tanh(self.online(state))
        elif model == "target":
            a = torch.tanh(self.target(state))
        #a = torch.tanh(self.online(state))
        mu = self.mu_head(a)
        sigma = F.softplus(self.sigma_head(a)) + 0.1

        return mu, sigma

    def get_dist(self, state, rate, model):
        mu, sigma = self.forward(state, model)
        dist = Normal(mu, sigma*rate)
        return dist

class PPO_Critic(TTENet):
    def __init__(self, state_dim, output_dim):
        super(PPO_Critic, self).__init__(state_dim, output_dim)

        self.C1 = nn.Linear(output_dim, 1)

    def forward(self, state, model):
        v = torch.tanh(self.online(state))
        v = self.C1(v)
        return v

class TTE_MSAN:
    def __init__(self, state_dim, action_dim, save_dir, is_train,use_cuda):
        """
        User input:
        The shapes of states and actions: state_dim,action_dim
        Storage directory: save_dir
        Definition within the function:
        Whether CUDA: use_cuda
        Policy Network: net
        Explore probability and decay patterns: exploration_rate,exploration_rate_decay,exploration_rate_min
        Time step count: curr_step
                Model storage interval: save_every Note: After initializing the neural network, it needs to be passed to the GPU.
        """
        #super().__init__(state_dim, action_dim, save_dir)
        self.memory = deque(maxlen=100000)
        self.memoryppo = deque(maxlen=10000)
        self.memoryok = deque(maxlen=500000)
        self.memorymax = deque(maxlen=500000)

        self.cache = deque(maxlen=50000)
        self.cacheM = deque(maxlen=50000)
        self.maxreaward = [0] *100000
        self.maxnum = 0

        self.K_epochs = 5  # PPO fragmentation update times
        self.entropy_coef = 1e-3  # the entropy coefficient of PPOctor
        self.entropy_coef_decay = 0.99  # PPOentropy_coef decay rate
        self.l2_reg = 1e-3  # the L2 regularization coefficient of PPOCritic
        self.clip_rate = 0.2  # PPO Clip rate
        self.batch_size = 32  # number of offline learning
        self.gamma = 0.99  # reward attenuation coefficient
        self.tau = 0.001
        self.lambd = 0.9  # GAE Attention factor
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.use_cuda = use_cuda
        self.is_train = is_train

        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # Action and Critic Model
        self.Mactor = DDPG_Actor(state_dim, action_dim).to(self.device)
        self.Mactor_target = DDPG_Actor(state_dim, action_dim).to(self.device)
        self.Mcritic = DDPG_Critic(state_dim, action_dim).to(self.device)
        self.Mcritic_target = DDPG_Critic(state_dim, action_dim).to(self.device)
        self.PPOactor = GaussianActor_musigma(state_dim, action_dim, 1).to(self.device)
        self.PPOcritic = PPO_Critic(state_dim, action_dim).to(self.device)
        self.DDPGactor = DDPG_Actor(state_dim, action_dim).to(self.device)
        self.DDPGactor_target = DDPG_Actor(state_dim, action_dim).to(self.device)
        self.DDPGcritic = DDPG_Critic(state_dim, action_dim).to(self.device)
        self.DDPGcritic_target = DDPG_Critic(state_dim, action_dim).to(self.device)
        self.DPGnet = DPG(self.state_dim, self.action_dim).to(self.device)
        self.DQNnet = DQN(self.state_dim, self.action_dim).to(self.device)
        self.DDQNnet = DDQN(self.state_dim, self.action_dim).to(self.device)
        self.DuelingDQNnet = DulineDQN(self.state_dim, self.action_dim).to(self.device)

        # Optimizer
        self.PPOactor_optimizer = torch.optim.Adam(self.PPOactor.parameters(), lr=0.00025)
        self.PPOcritic_optimizer = torch.optim.Adam(self.PPOcritic.parameters(), lr=0.00025)
        self.DDPGactor_optim = torch.optim.Adam(self.DDPGactor.parameters(), lr=0.00025)
        self.DDPGcritic_optim = torch.optim.Adam(self.DDPGcritic.parameters(), lr=0.00025)
        self.Mactor_optim = torch.optim.Adam(self.Mactor.parameters(), lr=0.00025)
        self.Mcritic_optim = torch.optim.Adam(self.Mcritic.parameters(), lr=0.00025)
        self.DPGoptimizer = torch.optim.Adam(self.DPGnet.parameters(), lr=0.00025)
        self.DQNoptimizer = torch.optim.Adam(self.DQNnet.parameters(), lr=0.00025)
        self.DDQNoptimizer = torch.optim.Adam(self.DDQNnet.parameters(), lr=0.00025)
        self.DuelingDQNoptimizer = torch.optim.Adam(self.DuelingDQNnet.parameters(), lr=0.00025)

        # Loss function
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.RLU_fn = torch.nn.ReLU()

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999975
        self.exploration_rate_min = 0.1 # min exploration probability
        self.curr_step = 0 # timestep count
        self.save_every = 5e5  # no. of experiences between saving Mario Net
        self.noisy = Normal(1, 3)
        self.burnin = 1000  # Experience pool accumulation: burnin, no learning until accumulation reaches this value.
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e3  # no. of experiences between Q_target & Q_online sync
        self.rotation = 0 # Loop consistency for act
        self.rotationlearn = 0  # Loop consistency for learn

    def get_dist_action(self, state):  # Get the random probability model in ppo
        with torch.no_grad():
            dist = self.PPOactor.get_dist(state, self.exploration_rate, "online")
            a = dist.sample()
            logprob_a = dist.log_prob(a).cpu().numpy().flatten()
            return a, logprob_a

    def act(self, state, period):
        """
            exploit: to select the optimal action; epsilon-greedy
            explore: Randomly select an action. The inputs and outputs of act are:
            Input: state, note: the state parameter state also needs to be passed to the GPU;
            Output: Action index action_idx Also, the timestep count parameter curr_step is +1 for each action.
        """

        now_action = [0, 0, 0, 0, 0]
        state = state.__array__()
        if self.use_cuda:
            state = torch.tensor(state).cuda()
        else:
            state = torch.tensor(state)
        state = state.unsqueeze(0)
        with torch.no_grad():
            DDPGaction_values1 = self.DDPGactor(
                state).squeeze() + self.is_train * self.exploration_rate * self.noisy.sample().squeeze(0)
            PPOaction_values, logprob_a = self.get_dist_action(state)  # PPO
            DPGvalues = self.DPGnet(state, model="online")  # DPG
            DPGaction_values = torch.tensor(
                [np.random.choice(range(DPGvalues.shape[1]), p=DPGvalues.data.cpu().flatten().numpy())])
            DQNaction_values = self.DQNnet(state, model="online")  # DQN
            DDQNaction_values = self.DDQNnet(state, model="online")  # DDQN
            action_all = [DDPGaction_values1, PPOaction_values.squeeze(0).squeeze(0), DPGaction_values.squeeze(0),
                          DQNaction_values, DDQNaction_values]  # concat action
            if np.random.rand() < self.exploration_rate:  # rand
                ran = np.random.randint(0, period-1, size=(1, 2))[0]
                action_all[3] = torch.tensor([ran[0]])
                action_all[4] = torch.tensor([ran[1]])
            else:
                action_all[3] = torch.argmax(DQNaction_values[:period], axis=1).squeeze(
                    0)
                action_all[4] = torch.argmax(DDQNaction_values[:period], axis=1).squeeze(
                    0)
            for i in range(len(action_all)):
                now_action[i] = round(torch.clamp(action_all[i], 0, period - 1).cpu().numpy().flatten()[0])

        return action_all, np.array(now_action), logprob_a.squeeze(0)

    def actM(self, state, period):
        state = state.__array__()
        if self.use_cuda:
            state = torch.tensor(state).cuda()
        else:
            state = torch.tensor(state)
        state = state.unsqueeze(0)
        action_idx = self.Mactor(state).squeeze()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # # increment step
        self.curr_step += 1
        return action_idx

    def caching(self, state, next_state, a, action, reward, done, dw, logprob_a,x_num):
        """
        Store the experience to self.cache (replay buffer)
        Inputs:
        state (matrix),
        next_state (matrix),
        action (int),
        reward (float),
        done(2,0,1))
        cache() - Stores memory: Every time the agent performs an action, it stores the experience generated by this action in its memory. The experience includes the current state, action, reward, next state, and whether the game has ended (done) *. The input and output of the cache function are respectively:
        Input: status, next status, action, reward, done;
        Output: Updated experience replay pool.
        """
        state = state.__array__()
        next_state = next_state.__array__()
        self.cache.append((state, next_state, a, action, reward, done,dw, logprob_a, x_num))

    def Mcaching(self, state, next_state, a, action, reward, done, dw,
                                             logprob_a,x_num):
        state = state.__array__()
        next_state = next_state.__array__()
        self.cacheM.append((state, next_state, a, action, reward, done,dw, logprob_a,x_num))

    def store(self, data, flag = 0):
        """
        Handle the R value feedback and store the cached data in memory
        """
        self.siga = 0.6  # Discount rate
        #flag = 0
        nowrew = 0
        while True:
            try:
                (state, next_state, a, action, reward, done, dw, logprob_a, x_num) = data.pop() # Pop the element from the far right
                if self.maxreaward[0]<reward and self.maxnum<x_num:
                    self.maxreaward[0] = reward
                    self.maxnum=x_num
                    #if x_num>len():
                    self.memoryok.clear()
                    flag = 1
                # if self.maxreaward[x_num]<reward:
                #     self.maxreaward[x_num] = reward
                #     flag = 2
                # if flag:
                #    reward = reward + self.siga*nowrew
                #    nowrew = reward

                state = torch.tensor(state)
                next_state = torch.tensor(next_state)
                a = torch.tensor([a])
                action = torch.tensor([action])
                reward = torch.tensor([reward]).float()
                done = torch.tensor([done])
                dw = torch.tensor([dw])
                logprob_a = torch.tensor(logprob_a)
                self.memory.append((state, next_state, a, action, reward, done, dw, logprob_a,))
                self.memoryppo.append(
                    (state, next_state, a, action, reward, done, dw, logprob_a,))  # Save ppo online solutions
                if flag:
                    self.memoryok.append((state, next_state, a, action, reward, done, dw, logprob_a,))  # Save excellent solutions

            except Exception as e:
                print(e)
                break
        data.clear()  # Clear Cache

    def recall(self):
        """
            Retrieve a batch of experiences from memory
            rechall() - Retrieving Memory: When an agent is learning and updating the network, it needs to retrieve a batch of experiences from the memory. The memory extraction uses the random.sanple() function to extract a certain number of samples from the given sequence to form a list. The input and output of the recall() function are respectively:
            Input: None
            Output: The status of one batch, the next status, action, reward, done (each encapsulated).
        """
        if len(self.memoryok) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            batch2 = random.sample(self.memoryok, self.batch_size)
            batch = batch+batch2
        else:
            batch = random.sample(self.memory, self.batch_size*2)
        state, next_state, a, action, reward, done, dw, logprob_a = map(torch.stack, zip(*batch))
        #self.memory.clear()
        return state.to(self.device), next_state.to(self.device), a.to(self.device), action.to(self.device).long(), reward.to(self.device), \
               done.to(self.device), dw.to(self.device), logprob_a

    def recallM(self):
        """
        Retrieve a batch of experiences from memory for Master.
        """
        if len(self.memoryok) > self.batch_size*2:
            batch = random.sample(self.memoryok, self.batch_size*2)
            #batch = batch+batch2
        else:
            batch = random.sample(self.memory, self.batch_size*2)

        state, next_state, a, action, reward, done, dw, logprob_a = map(torch.stack, zip(*batch))
        return state.to(self.device), next_state.to(self.device), a.to(self.device), action.to(self.device), reward.to(self.device),\
               done.to(self.device), dw.to(self.device), logprob_a

    def recall_all(self):
        """
        Retrieve a batch of experiences from memory for dom.
        """
        batch = random.sample(self.memoryppo, min([len(self.memoryppo), 1000]))
        state, next_state, a, action, reward, done, dw, logprob_a = map(torch.stack, zip(*batch))
        self.memoryppo.clear()
        return state.to(self.device), next_state.to(self.device), a.to(self.device), action.to(self.device), reward.to(self.device), \
               done.to(self.device), dw.to(self.device), logprob_a.to(self.device)


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param * (1.0 - tau) + param * tau)

    def sync_Q_target(self):  # Update the target network
        self.DQNnet.target.load_state_dict(self.DQNnet.online.state_dict())
        self.DDQNnet.target.load_state_dict(self.DDQNnet.online.state_dict())
        self.DuelingDQNnet.target.load_state_dict(self.DuelingDQNnet.online.state_dict())

    def save(self, over):
        if over:
            save_path = (
                Path(str(self.save_dir) + "/Snet_model_over.chkpt")
            )
        else:
            save_path = (
                    Path(str(self.save_dir) + f"/Snet_model_{int(self.curr_step // self.save_every)}.chkpt")
            )
        torch.save(
            dict(actor_model=self.Mactor.state_dict(), critic_model=self.Mcritic.state_dict()
                 , exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, name):
        load_path = (
                name
        )
        model = torch.load(load_path)
        self.Mactor.load_state_dict(model["actor_model"])
        self.Mcritic.load_state_dict(model["critic_model"])
        self.exploration_rate = model["exploration_rate"]

    def calc_advantage(self, state, next_state, reward, done, dw):
        with torch.no_grad():
            vs = self.PPOcritic(state, "online")
            vs_next = self.PPOcritic(next_state, "online")
            deltas = (reward * self.gamma * (vs_next-vs).T).T
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]
            for dlt, mask in zip(deltas, done.cpu().flatten().numpy()):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
                adv.append(advantage)
            # adv.reverse()
            adv = copy.deepcopy(adv[1:])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
            td_target = torch.tensor(deltas).unsqueeze(1).float().to(self.device)
            adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  # sometimes helps
            return td_target, adv

    def td_action(self, a, action, s, logprob_a, adv, reward):
        loss = []
        batch_size = self.batch_size*2
        choice = random.randint(0, len(a)//batch_size)
        torch.cuda.empty_cache()
        index = slice(choice * batch_size, min((choice + 1) * batch_size, s.shape[0]))
        distribution = self.PPOactor.get_dist(s[index], self.exploration_rate, "online")
        dist_entropy = distribution.entropy().sum(1, keepdim=True)
        logprob_a_now = distribution.log_prob(a[index])[0, ...]

        ratio = torch.exp(logprob_a_now - logprob_a[index])  # a/b == exp(log(a)-log(b))

        surr1 = ratio * adv[index].T
        surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index].T
        action_inform, _ = self.PPOactor(s[index], model="online")

        a_loss = -torch.min(surr1, surr2) - self.entropy_coef*dist_entropy
        a_loss = 0.1 * a_loss.mean() + 0.9*F.mse_loss(action_inform[:, 0].squeeze(-1), action[index].float().squeeze(-1))
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        self.PPOactor_optimizer.zero_grad()
        a_loss.backward()
        self.PPOactor_optimizer.step()
        loss.append(a_loss)

        return torch.tensor(loss)

    def td_critis(self, s, td_target ,a):
        loss = []
        #batch_size = len(td_target) // (self.batch_size)
        batch_size = self.batch_size*2
        choice = random.randint(0, len(td_target)//batch_size)
        index = slice(choice * batch_size, min((choice + 1) * batch_size, s.shape[0]))
        c_loss = (self.Mcritic(s[index], a[index]) - td_target[index]).pow(2).mean()
        c_loss2 = (self.PPOcritic(s[index], model="online") - td_target[index]).pow(2).mean()
        for name, param in self.PPOcritic.named_parameters():
            if 'weight' in name:
                c_loss += param.pow(2).sum() * self.l2_reg
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        self.Mcritic_optim.zero_grad()
        self.PPOactor_optimizer.zero_grad()
        c_loss.backward()
        c_loss2.backward()
        self.Mcritic_optim.step()
        self.PPOactor_optimizer.step()
        loss.append(c_loss.mean())
        return torch.tensor(loss)

    def updataM(self, state, next_state, action, reward, done,dw):
        with torch.no_grad():
            Q_next = self.Mcritic_target(next_state, self.Mactor_target(
                next_state))
            #Q_next = self.Mcritic_target(state, a)
            #Q_target = self.gamma * torch.mul(reward, Q_next)
            #Q_target = reward + (1 - dw.float()) * self.gamma * Q_next
            Q_target = reward + (1 - dw.float()) * self.gamma * Q_next
        critic_loss = F.mse_loss(self.Mcritic(state, action.float()), Q_target)

        self.Mcritic_optim.zero_grad()
        critic_loss.backward()
        self.Mcritic_optim.step()

        # unpate actor
        self.Mcritic.eval()
        action_inform = self.Mactor(state)

        actor_loss = - 0.21 * self.Mcritic(state, action_inform) + 0.001 * F.mse_loss(action_inform,
                                                                                     action.float()) * (reward + (1 - dw.float()))

        # print(actor_loss.shape)
        actor_loss = actor_loss.mean()
        self.Mactor_optim.zero_grad()
        actor_loss.backward()
        self.Mactor_optim.step()
        self.Mcritic.train()


        return torch.mean(Q_target).item(), actor_loss.item()

    def updataDDPG(self, state, next_state, action, reward, done, dw):
        with torch.no_grad():
            Q_next = self.DDPGcritic(next_state, self.DDPGactor_target(
                next_state))
            #Q_target = (reward) * self.gamma * Q_next
            Q_target = reward + (1 - dw.float()) * self.gamma * Q_next
            #Q_target = (reward + (1 - done.float())) * self.gamma * Q_next

        critic_loss = F.mse_loss(self.DDPGcritic(state, action.float()), Q_target)

        self.DDPGcritic_optim.zero_grad()
        # critic_loss.backward()
        self.DDPGcritic_optim.step()

        # 更新actor
        self.DDPGcritic.eval()
        action_inform = self.DDPGactor(state)
        actor_loss = - self.DDPGcritic(state, action_inform)

        # print(actor_loss.shape)
        actor_loss = actor_loss.mean()
        self.DDPGactor_optim.zero_grad()
        actor_loss.backward()
        self.DDPGactor_optim.step()
        self.DDPGcritic.train()

        return torch.mean(Q_target).item(), actor_loss.item()

    def updataPPO(self, state, next_state, a, action, reward, done, dw, logprob_a):
        a_lossall = []
        c_lossall = []
        with torch.no_grad():
            self.entropy_coef *= self.entropy_coef_decay
            td_target, adv = self.calc_advantage(state, next_state, reward, done, dw)
        for i in range(1):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            with torch.no_grad():
                perm = np.arange(state.shape[0])
                np.random.shuffle(perm)
                perm = torch.LongTensor(perm).to(self.device)
                s, a2, action2, td_target2, adv2, lo_a, reward2 = \
                    state[perm].clone(), a[perm].clone(), action[perm].clone(), td_target[perm].clone(), adv[perm].clone(), \
                    logprob_a[perm].clone(), reward[perm].clone()  # 微小样本
            '''update the actor'''
            a_loss = self.td_action(a2, action2, s, lo_a, adv2, reward2)
            a_lossall.append(torch.mean(a_loss).item())
            '''update the critic'''
            c_loss = self.td_critis(s, td_target2,a2)
            c_lossall.append(torch.mean(c_loss).item())
        return torch.mean(td_target).item(), np.mean(a_lossall)

    def updataPG(self, state, action, reward, done):
        softmax_input = self.DPGnet.forward(state, "online")
        action= action.long().squeeze()
        neg_log_prob = F.cross_entropy(input=softmax_input, target=action,
                                       reduction='none')  # 交叉熵---当前Policy Gradient网络计算出的行为与实行的行为之间的差异

        # Step 3: back
        loss = torch.mean(neg_log_prob * (reward + (1 - done.float())))
        self.DPGoptimizer.zero_grad()
        loss.backward()
        self.DPGoptimizer.step()
        return torch.mean(neg_log_prob).item(), loss.item()

    def updataDDQN(self, state, next_state, action, reward, done):
        td_est = self.DDQNnet(state, model="online")[
            np.arange(0, state.shape[0]), action
        ].squeeze(-1)  # Q_online(s,a)
        next_state_Q = self.DDQNnet(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.DDQNnet(next_state, model="target")[
                np.arange(0, state.shape[0]), best_action
            ]
        td_tgt = (reward + (1 - done.float()) * self.gamma * next_Q).float().squeeze(-1)
        loss = self.loss_fn(td_est, td_tgt)
        self.DDQNoptimizer.zero_grad()
        loss.backward()
        self.DDQNoptimizer.step()
        return td_est.mean().item(), loss.item()

    def updataDQN(self, state, next_state, action, reward, done):
        td_est = self.DQNnet(state, model="online")[
            np.arange(0, state.shape[0]), action
        ].squeeze(-1)  # Q_online(s,a)
        next_state_Q = self.DQNnet(next_state, model="target")
        next_Q = next_state_Q[
            np.arange(0, state.shape[0]), action
        ]  # Q_target(s,a)
        td_tgt = (reward + (1 - done.float()) * self.gamma * next_Q).float().squeeze(-1)
        loss = self.loss_fn(td_est, td_tgt)
        self.DQNoptimizer.zero_grad()
        loss.backward()
        self.DQNoptimizer.step()
        return td_est.mean().item(), loss.item()

    def updataDuelingDQN(self, state, next_state, action, reward, done):
        V, A = self.DuelingDQNnet(state, model="online")
        td_est = (V + A - torch.mean(A, dim=1, keepdim=True))[
            np.arange(0, state.shape[0]), action
        ].squeeze(-1)  # Q_online(s,a)
        V_est, next_state_Q = self.DuelingDQNnet(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        V_, A_ = self.DuelingDQNnet(next_state, model="target")
        next_Q = (V_ + A_ - torch.mean(A_, dim=1, keepdim=True))[
            np.arange(0, state.shape[0]), best_action
        ]# Q_target(s,a)
        td_tgt = (reward + (1 - done.float()) * self.gamma * next_Q).float().squeeze(-1)
        loss = self.loss_fn(td_est, td_tgt)
        self.DuelingDQNoptimizer.zero_grad()
        loss.backward()
        self.DuelingDQNoptimizer.step()
        return td_est.mean().item(), loss.item()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
            self.soft_update(self.Mactor_target, self.Mactor, self.tau)
            self.soft_update(self.Mcritic_target, self.Mcritic, self.tau)
            self.soft_update(self.DDPGactor_target, self.DDPGactor, self.tau)
            self.soft_update(self.DDPGcritic_target, self.DDPGcritic, self.tau)
        if self.curr_step % self.save_every == 0:
            self.save(0)  # Store network parameter information,
        if self.curr_step < self.burnin or len(self.memory) < self.burnin:# If the stored information is limited
            return [0,0,0,0,0,0], [0,0,0,0,0,0]
        if self.curr_step % self.learn_every != 0:
            return [0,0,0,0,0,0], [0,0,0,0,0,0]
        # Sample from memory
        stateM, next_stateM, aM, actionM, rewardM, doneM,dwM, logprob_aM = self.recallM()

        q1, aloss1 = self.updataM(stateM, next_stateM, aM, rewardM, doneM, dwM)
        state, next_state, a, action, reward, done, dw, logprob_a = self.recall()

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        #try:
        q2, aloss2 = 0, 0
        q3, aloss3 = 0, 0
        q4, aloss4 = 0, 0
        q5, aloss5 = 0, 0
        q6, aloss6 = 0, 0
        if self.rotationlearn %6 ==0:
            q2, aloss2 = self.updataDDPG(state, next_state, a, reward, done, dw)
        elif self.rotationlearn %6<=2 and len(self.memoryppo)>1000:  # 单次交易
            state, next_state, a, action, reward, done, dw, logprob_a = self.recall_all()
            q3, aloss3 = self.updataPPO(state, next_state, a, action, reward, done, dw, logprob_a)
            q4, aloss4 = self.updataPG(state, action, reward, done)

        elif self.rotationlearn %6 == 3:
            q5, aloss5 = self.updataDQN(state, next_state, action, reward, done)
        elif self.rotationlearn %6 == 4:
            q6, aloss6 = self.updataDDQN(state, next_state, action, reward, done)
        elif self.rotationlearn%6 == 5:
            closs2, aloss2 = self.updataDuelingDQN(state, next_state, action, reward, done)
        #except Exception as e:
        #    q2, aloss2 = q1, aloss1

        self.rotationlearn += 1

        return [round(q2, 4),round(q3, 4),round(q4, 4),round(q5, 4),round(q6, 4), round(q1, 4)], \
               [round(aloss2, 4), round(aloss3, 4), round(aloss4, 4), round(aloss5, 4), round(aloss6, 4), round(aloss1, 4)]

