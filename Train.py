from Import import *
from Agent import Agent
from Env import Env
from collections import deque
from utils import get_padding_sequence,custom_loss_function

class Train:
    def __init__(self):
        self.sample_batch_size = 128
        self.episodes          = 10000
        self.state_size        = 150
        self.action_size       = 3
        self.agent             = Agent(self.state_size, self.action_size)
        self.env = Env([0,1,2],[j for j in range(1,self.state_size)],self.agent.x_train[0], self.agent.y_train[0])
        self.label_data = 1.0
        self.label_past = 1.0
    def run(self):
        try:
            for index_episode in range(self.episodes):
                #time_to_begin = random.randint(1,self.state_size)
                index_random_data = random.randint(0,49)
                seq = self.agent.x_train[index_random_data]
                seq_label = self.agent.y_train[index_random_data]
                self.env.reset(seq,seq_label)
                done = False
                index = 1 #time_to_begin
                while not done and index<=150:
                    state = self.env.get_sequence_state()
                    action = self.agent.act(state,index)
                    next_state, reward, done = self.env.step(action)
                    next_state = self.env.get_sequence_state()       
                    next_state = np.reshape(self.env.get_sequence_state(),(1,150,1,1))
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index+=1
                if(index_episode%20==0):
                    print("Episode {}".format(index_episode))
                if index_episode % 100==0 and index_episode != 0:
                    acc,res,t = self.agent.compute_acc()
                    acc_val,res_val,t_val = self.agent.compute_acc_val()
                    print("acc_train {} ======> average_time_train {} ======> update {}".format(acc, np.mean(t), self.agent.update_number))
                    print("acc_val {} ======> average_time_val {} ======> update {}".format(acc_val, np.mean(t_val), self.agent.update_number)) 
                    if acc > 0.9 :
                        self.agent.save_weight()
                       
                self.agent.replay(self.sample_batch_size)
                self.agent.target_train()
        finally:
            self.agent.save_model()
if __name__ == "__main__":
    train = Train()
    train.run()
    