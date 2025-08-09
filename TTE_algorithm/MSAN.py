import copy

#import numba.cuda
import time
import psutil
import package as package
from pathlib import Path
import datetime
import torch
import json
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
class MSAN_SAgent():
    def __init__(self, filepath, file1, gap, ttype, leng, is_train=True):
        file = file1.split('.')[0]
        messages, smessages, self.links, self.nodes = package.Prase_Prj_to_dict.init_messages(filepath+file1)
        self.ndata = []
        [self.ndata.append(messages[m]) for m in messages]
        self.starttime = time.time()
        self.data = self.get_date(self.ndata, leng)

        self.gap = gap
        self.CC = package.get_cc(self.data, gap=float(gap))   # 集群周期
        self.pEenv, self.mEenv = package.build_environment(self.data, self.links, self.CC, gap=float(gap))  # 建立基本环境
        self.haveresult, smess_leng = package.get_result(self.data, self.links, self.CC)   # 检查是否有解
        self.max_locepisodes = 100
        print("Initialization is complete~")
        if not self.haveresult:
            self.TTE_Env = package.TTE_Env_SAgent.TTE(self.data, self.links, self.CC, gap)  # 智能体Env环境---单智能体
            self.TTE_EnvM = package.TTE_Env_SAgent.TTE(self.data, self.links, self.CC, gap)  # 智能体Env环境---单智能体
            self.save_dir = Path("checkpoints2/SAgent/" + "MSAN/" + file) / ("_"+gap+datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))

            self.ttype = ttype
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")

            self.tabel = "_MSAN_"+str(leng)
            print(f"Using CUDA: {use_cuda}")
            self.sava_modeidrnoTD = Path(str(self.save_dir) + "_model"+ self.tabel)
            sava_modeidr = self.sava_modeidrnoTD
            sava_modeidr.mkdir(parents=True)
            self.maxreward = 0
            self.starttime = time.time()
            self.TTE_Solver = package.MSAN_Solver_SAgent.TTE_MSAN(state_dim=(1, 4, self.CC),
                                                                   action_dim=self.TTE_Env.action_space.n, save_dir=self.sava_modeidrnoTD, is_train=is_train, use_cuda=use_cuda)
            self.logger = package.Logger.MetricLogger(str(self.save_dir)+self.tabel, self.max_locepisodes)  #

    def get_date(self, ndata, leng):
        data = copy.deepcopy(ndata)
        if len(data) > leng:
            data = np.array(data)
            bbb = list(range(leng))
            np.random.seed(leng)  # 设置随机种子为123
            datanew = data[np.random.choice(bbb, size=leng, replace=False)]
        else:
            datanew = []
            for i in range(leng):
                kk = np.random.randint(0, len(data))
                datanew.append(copy.deepcopy(data[kk]))
        msid = 0
        for i in range(len(datanew)):
            datanew[i]["msgid"] = msid
            msid += 1
        return datanew

    def softmax(self, X):
        X_exp = np.exp(X)
        partition = X_exp.sum()
        return X_exp / partition

    def complex_action(self, actionall):
        max_reward = -99
        max_ai = 0
        for ai in range(len(actionall)):
            now_Env = copy.deepcopy(self.TTE_Env)
            _, now_reward, _, _ = now_Env.step(actionall[ai])
            if now_reward > max_reward:
                max_ai = ai
                max_reward = now_reward
        return max_ai

    def get_Q(self, state, actions):
        with torch.no_grad():
            state2 = copy.deepcopy(state).unsqueeze(0).to(self.device).repeat_interleave(len(actions), dim=0)
            new_as = self.TTE_Solver.Mcritic(state2, torch.tensor(actions.reshape(len(actions), 1)).to(self.device))
        return new_as.data.cpu().flatten().numpy()

    def learn(self):
        episodes = 10000
        # TTE_Solver.load("F:/Work/Deep_Rein/TTE/checkpoints/SAgent/Ladder/L2_15.625/2022-08-03T20-56-15_model/Snet_model_over.chkpt")
        Rmax= 0
        n_num = 0
        flag = 0
        rotion = [0,0,0,0,0,0,0]
        maxrate = 0
        maxfram = 0
        max_x = 0
        self.maxreward = 0
        max_locepisodes = self.max_locepisodes
        polling = 0
        for e in range(episodes):
            state = self.TTE_Env.reset()
            state = self.TTE_EnvM.reset()

            # Play the game!
            locepisodes = max_locepisodes
            if self.TTE_Solver.rotation > 1e8:
                self.TTE_Solver.rotation = 0
            x_num = 0
            q,loss = [0,0,0,0,0,0], [0,0,0,0,0,0]
            #polling = 0
            while locepisodes > 0:
                # state = TTE_Env.reset()
                # Run agent on the state
                period = self.TTE_Env.message[self.TTE_Env.nowstate[0]]["period"]
                acall, actionall, logprob_a = self.TTE_Solver.act(state, period)
                aM = self.TTE_Solver.actM(state, period)

                actionM = round(torch.clamp(aM, 0, period-1).item())
                aci = actionall[polling%5]

                idx = polling%5
                next_state, reward, done, info = self.TTE_Env.step(aci)

                next_stateM, rewardM, doneM, infoM = self.TTE_EnvM.step(actionM)

                # Update state
                if reward < rewardM:
                #if reward < rewardM:
                    rotion[-1] += 1
                    done = doneM
                    info = infoM
                    if done and info["flag_get"] != 2:
                        self.TTE_Solver.Mcaching(state, next_stateM, aM, actionM, rewardM, doneM, 1,
                                                 logprob_a, x_num)
                    else:
                        self.TTE_Solver.Mcaching(state, next_stateM, aM, actionM, rewardM, doneM, 0,
                                                 logprob_a, x_num)
                    state = next_stateM
                    #print("cache Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")
                    self.TTE_Env.mEenv = copy.deepcopy(self.TTE_EnvM.mEenv)
                    self.TTE_Env.pEenv = copy.deepcopy(self.TTE_EnvM.pEenv)
                    self.TTE_Env.nowstate = self.TTE_EnvM.nowstate
                    reward = rewardM

                else:
                    rotion[idx] += 1
                    if done and info["flag_get"] != 2:
                        self.TTE_Solver.caching(state, next_state, acall[idx], aci, reward, done, 1, logprob_a, x_num)
                        self.TTE_Solver.Mcaching(state, next_state, acall[idx], aci, reward, done, 1, logprob_a, x_num)
                        #self.maxreward = reward
                    else:
                        self.TTE_Solver.caching(state, next_state, acall[idx], aci, reward, done, 0, logprob_a,x_num)
                        self.TTE_Solver.Mcaching(state, next_state, acall[idx], aci, reward, done, 0, logprob_a,x_num)
                    if reward >= self.maxreward:
                        self.maxreward = reward

                    state = next_state
                    self.TTE_EnvM.mEenv = copy.deepcopy(self.TTE_Env.mEenv)
                    self.TTE_EnvM.pEenv = copy.deepcopy(self.TTE_Env.pEenv)
                    self.TTE_EnvM.nowstate = self.TTE_Env.nowstate


                # Check if end of game
                if done or info["flag_get"] == 2:
                    locepisodes -= 1

                    self.TTE_Solver.stopstate = self.TTE_Env.stopstate
                    self.TTE_Solver.store(self.TTE_Solver.cache)
                    #if len(self.TTE_Solver.memoryok)< 30000:
                    self.TTE_Solver.store(self.TTE_Solver.cacheM, flag=1)
                    #else:
                    #    self.TTE_Solver.store(self.TTE_Solver.cacheM, flag=0)

                    state2 = self.TTE_Env.reset()
                    state = self.TTE_EnvM.reset()  # 环境转换
                    # Learn
                    x_num = 0

                    # Logging9
                    if n_num < 100:
                        q, loss = self.TTE_Solver.learn()
                        self.logger.log_step(reward, loss, q)  # 记录回报、损失和Q值
                    #else:
                    #    self.logger.log_step(reward, 0, 0)  # 记录回报、损失和Q值
                    self.logger.log_episode()
                    if info["flag_get"] == 2:
                        Rmax += 1
                        #break
                    else:
                        polling += 1
                        #self.TTE_Solver.rotation += 1  # 均是失败的调度才换轮询
                else:
                    x_num+=1
                    #q, loss = self.TTE_Solver.learn()
                    self.logger.log_step(reward, [0,0,0,0,0,0], [0,0,0,0,0,0])  # 记录中间回报、损失和Q值
            #
            if Rmax>=max_locepisodes:  # 找到可行解就退出
                n_num +=1
            else:
                n_num = 0
            #if n_num > 100: # convergence
            #    break
            #    self.TTE_Solver.save(1)

            if e % 1 == 0:
                self.logger.record(episode=e, epsilon=self.TTE_Solver.exploration_rate, step=self.TTE_Solver.curr_step, resultnum=Rmax,
                                   DDPGn = rotion[0], PPOn=rotion[1], DPGnn=rotion[2],DQNn=rotion[3], DDQNn=rotion[4],MSANn=rotion[-1], Loss=loss, Qs=q)
                Rmax = 0
                rotion = [0, 0, 0, 0, 0, 0, 0]
            if e % 100 == 0:
                print(rotion, max_x, self.TTE_Solver.maxreaward[max_x], len(self.TTE_Solver.memoryok), len(self.TTE_Solver.memorymax), maxrate, maxfram)

        self.TTE_Solver.save(1)

    def show(self, name):
        check_a = package.check_result(self.data, self.pEenv, self.mEenv, copy.deepcopy(self.TTE_Env.return_dict_over))
        pEenvn, mEenvn, df = package.mark_location_in_environment(self.data, self.pEenv, self.mEenv, copy.deepcopy(self.TTE_Env.return_dict_over))
        package.plot_link(df, self.links, name)
        pEenvn, mEenvn, df = package.mark_location_in_environment(self.data, self.pEenv, self.mEenv,
                                                                  copy.deepcopy(self.TTE_Env.return_dict_over))
        package.plot_link(df, self.links, name)

