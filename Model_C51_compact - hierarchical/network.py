import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random

class experience_buffer():
    def __init__(self, buffer_size=5000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def cut(self,length):
        #discard a batch of experience recorded, onluy keep -1-length:-1 buffers
        if len(self.buffer)>length:
            self.buffer=self.buffer[-1-length:]

    def sample(self, batch_size, trace_length):
        sampled_episodes=[]
        for i in range(batch_size):
            sampled_episodes.append(random.choice(self.buffer))
        sampledTraces = []
        for episode in sampled_episodes:
            sampledTraces.append(episode)
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 10])

class bandit_buffer():

        #each buffer is [feature, action, reward]
        def __init__(self, buffer_size=5000):
            self.buffer = []
            self.buffer_size = buffer_size

        def add(self, experience):
            if len(self.buffer) + 1 >= self.buffer_size:
                self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
            self.buffer.append(experience)

        def cut(self, length):
            # discard a batch of experience recorded, onluy keep -1-length:-1 buffers
            if len(self.buffer) > length:
                self.buffer = self.buffer[-1 - length:]

        def recent_sample(self,batch_size):
            sampled_episodes = []
            samplesize=max(len(self.buffer)//2,2000)
            for i in range(batch_size):
                sampled_episodes.append(random.choice(self.buffer[-samplesize:]))
            sampledTraces = []
            for episode in sampled_episodes:
                sampledTraces.append(episode)
            sampledTraces = np.array(sampledTraces)
            return np.reshape(sampledTraces, [batch_size, 3])


        def sample(self, batch_size):
            sampled_episodes = []
            for i in range(batch_size):
                sampled_episodes.append(random.choice(self.buffer))
            sampledTraces = []
            for episode in sampled_episodes:
                sampledTraces.append(episode)
            sampledTraces = np.array(sampledTraces)
            return np.reshape(sampledTraces, [batch_size, 3])


#functions
def updateTarget(op_holder,sess):
    sess.run(op_holder)

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value()))) #tau*main_network + (1-tau)*target network, soft update
    return op_holder


def processState(state,Nstation):
    #input is the N by N by 6 tuple, map it to a list
    input_dim=5;
    return np.reshape(state,[Nstation*Nstation*input_dim])


def relocation_net_update(trainBatch,station):
    reward = np.array([r[station] for r in trainBatch[:, 2]])
    action = np.array([a[station] for a in trainBatch[:, 10]]) #last dimension

    return reward, action