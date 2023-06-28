import torch
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box

def ttf(np_arr):
    return torch.tensor(np_arr, dtype=torch.float, requires_grad=False)
def ttl(np_arr):
    return torch.tensor(np_arr, dtype=torch.long, requires_grad=False)

class Memory():
    def __init__(self, env, mem_size=2000, pt_dev='cpu'):
        self.env = env
        self.mem_size = mem_size
        self.pt_dev = pt_dev
        self.clear()
        
        
    def clear(self): # Will initialize/reset the memory
        self.obs_memory = torch.zeros((self.mem_size+1,)+self.env.observation_space.shape, 
                                      dtype=torch.float, device=self.pt_dev)
        self.final_memory = torch.zeros(self.mem_size+1, dtype=torch.long, device=self.pt_dev)
        self.reward_memory = torch.zeros(self.mem_size+1, dtype=torch.float, device=self.pt_dev)
                
        if type(self.env.action_space) is Discrete:
            self.action_memory = torch.zeros(self.mem_size, dtype=torch.long, device=self.pt_dev)
            self.info_memory = torch.zeros((self.mem_size, self.env.action_space.n), dtype=torch.float, device=self.pt_dev)
        elif type(self.env.action_space) is Box:
            self.action_memory = torch.zeros((self.mem_size, )+tuple(self.env.action_space.shape), 
                                             dtype=torch.long, device=self.pt_dev).squeeze()
            self.info_memory = torch.zeros((self.mem_size, )+tuple(self.env.action_space.shape), 
                                             dtype=torch.float, device=self.pt_dev).squeeze()
        else:
            raise TypeError("Action space type not supported!")
        
        self.m_i = 0 # current index in the memory
        self.m_n = 0 # number of elements in the memory
        self.t_i = 0 # total number of steps given
        
        
    def store(self, act_i, rw_i, obs_i2, end_state, info_i, obs_i1=None):
        if obs_i1 is not None:
            self.obs_memory[self.m_i,:] = obs_i1 # we store the first observation
        if len(self.action_memory.shape)==1:
            self.action_memory[self.m_i] = act_i # store action in memory
        else:
            self.action_memory[self.m_i,:] = act_i # store action in memory

        self.reward_memory[self.m_i+1] = rw_i # store reward in memory
        self.obs_memory[self.m_i+1,:] = obs_i2 # we store the next observation
        self.final_memory[self.m_i+1] = end_state # store end state in memory
        self.info_memory[self.m_i,:] = info_i # store info in memory


        self.m_n = min(self.m_n+1, self.mem_size) # update number of elements in memory, and index
        self.m_i = (self.m_i + 1) % self.mem_size
        self.t_i += 1
        
        
    def get_memory(self): # returns 6 arrays with o_t, a_t, r_t, o_t+1, t_t and info_t
        return self.obs_memory[:self.m_n,:], \
               self.action_memory[:self.m_n] if len(self.action_memory.shape) == 1 else self.action_memory[:self.m_n,:], \
               self.reward_memory[1:self.m_n+1], \
               self.obs_memory[1:self.m_n+1,:], \
               self.final_memory[1:self.m_n+1], \
               self.info_memory[:self.m_n] if len(self.info_memory.shape) == 1 else self.info_memory[:self.m_n,:]


class Agent(): 
    def __init__(self, env, policy, memory, pt_dev='cpu'):
        self.env = env # the environment
        self.p = policy # the policy the agent will follow
        self.mem = memory # number of frames that will fit in the memory
        self.pt_dev = pt_dev # device where we will store the information
        
        self.__iter__()
    
    def __iter__(self): # with these two functions we can iterate through the episode
        self.c_obs = ttf(self.env.reset()[0].copy()).to(self.pt_dev) # reset function returns the first observation
        self.is_end_state = False
        return self
    
    def __next__(self): # with these two functions we can iterate through the episode
        if self.is_end_state:
            raise StopIteration
        else:
            act_i, info_i = self.p(self.c_obs.unsqueeze(0)) # our policy returns actions and additional info (eg. q values)
            obs_i, rw_i, end_state, _, _ = self.env.step(act_i.cpu().numpy() if type(self.env.action_space) is Box else \
                (act_i if type(act_i) is int else act_i.item()))
            
            self.c_obs = ttf(obs_i.copy()).to(self.pt_dev)
            self.is_end_state = end_state
            
            return obs_i, act_i, rw_i, end_state, info_i    
    
    def play(self, max_steps=10000, render=False, episode=False, store=True): # will play a game and store results in memory
        if episode:
            self.__iter__() # we start the game iteration
        if store:
            first_obs = 0.0 + self.c_obs
        rews = [] # we store the list of rewards for returning it later
        
        ii = 0
        epis = []
        while ii < max_steps:
            if render:
                self.env.render()
            try:
                obs_i, act_i, rw_i, end_state, info_i = self.__next__()
                epis.append(0)
            except StopIteration:
                self.__iter__()
                epis.append(1)
                if episode:
                    break
                else:
                    continue
            rews.append(rw_i)
            
            if store:
                if ii==0:
                    self.mem.store(act_i, rw_i, self.c_obs, end_state, info_i, first_obs)
                else:
                    self.mem.store(act_i, rw_i, self.c_obs, end_state, info_i)
            ii += 1
        return rews, epis # returns the number of rewards and episodes that happened
    
    
    def sample_rewards(self, n=100, max_steps=10000):
        return np.array(list([sum(self.play(max_steps, episode=True, store=False)[0]) for i in range(n)]))