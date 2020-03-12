from utils import get_padding_sequence,custom_loss_function
from Import import *
class Env():
    def __init__(self, action, states, sequence, label):
        self.action = action
        self.states = states
        self.sequence = {"data" : sequence, "label" : label}
        self.state = 1
        self.done = False
        self.reward = 0
        self.param_lambda = 0.001
        self.param_p = 1/3

    def step(self, action):
        if action == 0:  
            if not self.done and self.state < len(self.sequence["data"]):
                self.reward = -(self.param_lambda * (self.state ** self.param_p))
                self.state = self.state + 1
                
        if action == 1:  
            if not self.done and self.state <= len(self.sequence["data"]):
                if self.sequence["label"] == 1:
                    self.reward = 1
                else:
                    self.reward = -1
            
            self.done = True
   
        if action == 2:  
            if not self.done and self.state <= len(self.sequence["data"]):
                if self.sequence["label"] == 2:
                    self.reward = 1
                else:
                    self.reward = -1
            self.done = True
     
        return self.state, self.reward, self.done
            
            
    def reset(self, sequence, label):
        self.sequence = {"data" : sequence, "label" : label}
        self.state = 1
        self.done = False      
        return self.state, self.done  
    
    def set_label(self,label):
        self.sequence["label"] = label
            
    def get_sequence_state(self):
        return get_padding_sequence(self.sequence["data"], self.state) 
    
    def get_initial_sequence(self):
        return self.sequence["data"]
    
    def get_state(self):
        return self.state
            
            
    def render(self, mode='human'):
        return 0
    def close (self):
        return 0