import torch
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from torch.utils import data
import copy



def ttf(np_arr):
    return torch.tensor(np_arr, dtype=torch.float, requires_grad=False)
def ttl(np_arr):
    return torch.tensor(np_arr, dtype=torch.long, requires_grad=False)


class QPolicy():
    def __init__(self, q, env_act_space, eps=0.1, gamma=0.99, pt_dev='cpu', rand_policy=None):
        self.eps = eps
        self.gamma = gamma
        self.env_act_space = env_act_space
        self.pt_dev = pt_dev
        
        self.q = q # the 'playing' network, will be used to get the policy and play
        self.tq = copy.deepcopy(q) # the target network, will be used to estimate the q_t
        #init_f(self.q) # init the weights of the q network
        self.opt1 = torch.optim.Adam(self.q.parameters(), lr=0.001, amsgrad=True, weight_decay=0.0)

        if rand_policy is None:
            self.rand_policy = lambda: (ttl(self.env_act_space.sample()).to(self.pt_dev), torch.zeros(self.env_act_space.n).to(self.pt_dev))
        else:
            self.rand_policy = rand_policy
        
    def __call__(self, obs): # epsilon-greedy policy to encourage exploration
        if np.random.rand() <= self.eps: 
            return self.rand_policy() # follow random policy 
        else:
            with torch.no_grad():
                q_val = self.q(obs) # follow policy given by Q
            return torch.argmax(q_val, dim=1), q_val # return action and q valuues
        
    def q_target(self, r1, t1, o2): # a slightly modified target to improve stability
        with torch.no_grad():
            t_out = r1 + (1 - t1).float() * (self.gamma*torch.gather(self.tq(o2), dim=1,
                                                  index=torch.argmax(self.q(o2), dim=1).unsqueeze(1)).squeeze())
        return t_out
    
    def copy_q_to(self, ctq): # a function to set the target network equal to the 'playing' network
        ctq.load_state_dict(self.q.state_dict()) 
        
    def train(self, oarot, epochs=20, mini_b_size=None): # trains the 'playing' network
        if mini_b_size is not None:
            o1, a1, r1, o2, t1, _ = oarot
            self.q = self.q.to('cpu')
            self.tq = self.tq.to('cpu')
            q1 = self.q_target(r1, t1, o2)

            self.q = self.q.to(self.pt_dev)
            self.tq = self.tq.to(self.pt_dev)

            mini_b = data.DataLoader(data.TensorDataset(o1, a1, q1), batch_size=mini_b_size, shuffle=True)
            losses = sum([[self.train_step(oi.to(self.pt_dev), ai.to(self.pt_dev), qi.to(self.pt_dev)) for oi, ai, qi in mini_b] for ei in range(epochs)],[])
        else:
            o1, a1, r1, o2, t1, _ = [vi.to(self.pt_dev) for vi in oarot]
            q1 = self.q_target(r1, t1, o2)
            losses = [self.train_step(o1, a1, q1) for ep in range(epochs)]
        return losses
    
    def train_step(self, o1, a1, q1):
        self.opt1.zero_grad()
        loss = torch.nn.functional.mse_loss(torch.gather(self.q(o1), dim=1, index=a1.unsqueeze(1)).squeeze(), q1) #dqn loss
        l_out = loss.item() # we keep the loss value for output
        loss.backward() # we backpropagate the gradients with respect to our parameters
        self.opt1.step()
        return l_out