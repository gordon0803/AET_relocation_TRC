##function to assist the main system
import config
import numpy as np
import taxi_env
import pickle
from system_tracker import system_tracker
import bandit
import network
import os
import tensorflow as tf
import DRQN_agent

class env_agent():
    # ------------------Parameter setting-----------------------

    def __init__(self):
        self.load_params()
        self.env=taxi_env.taxi_simulator(self.arrival_rate,self.OD_mat,self.distance,self.travel_time,self.taxi_input)
        self.env.reset()
        self.sys_tracker = system_tracker()
        self.sys_tracker.initialize(config, self.distance, self.travel_time, self.arrival_rate, int(self.taxi_input), self.N_station, self.num_episodes,
                               self.max_epLength)
        self.tau=0.05 #frequency for target net update
        # process a new distance to avoid non zeros divison
        for i in range(self.N_station):
            self.distance[i, i] = 1

        # Set the rate of random action decrease.
        self.e = self.startE
        self.bandit_swap_e=self.e
        self.stepDrop = (self.startE-self.endE)/self.anneling_steps

        # create lists to contain total rewards and steps per episode
        self.jList = []
        self.rList = []
        self.total_steps = 0

        self.total_train_iter = 1;

        # Make a path for our model to be saved in.
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.linucb_agent = bandit.linucb_agent(self.N_station, self.N_station * 4)
        self.exp_replay = network.experience_buffer(5000)  # a single buffer holds everything
        self.bandit_buffer = network.bandit_buffer(5000)
        self.bandit_swap_e = 1;
        self.linucb_agent_backup = bandit.linucb_agent(self.N_station, self.N_station * 4)
        print('System Successfully Initialized!')

    def create_session(self):
        # force on gpu
        config1 = tf.ConfigProto()
        config1.gpu_options.allow_growth = True
        self.sess=tf.Session(config=config1)

    def create_stand_agent(self):
        self.agent = DRQN_agent.drqn_agent_efficient(self.N_station, self.h_size, self.lstm_units, self.tau, self.sess, self.batch_size, self.trace_length,
                                                is_gpu=self.use_gpu)

        self.agent.drqn_build()

    def initialize_episode(self):
        #buffer to be used within each episode
        self.global_epi_buffer=[]
        self.global_bandit_buffer=[]
        self.sys_tracker.new_episode() #tracker new episode
        self.env.reset() #taxi system reset
        self.init_state_list=[]
        if self.bandit_swap_e - self.e >.05:  # we do swapping when $e$ got declined by 0.05 percent.
            newb=bandit.linucb_agent(self.N_station,self.N_station*4)
            for st in range(self.N_station):
                newb.Aa[st]=0.1*np.identity(self.N_station*4)+self.linucb_agent_backup.Aa[st]
                newb.ba[st]=self.linucb_agent_backup.ba[st]
            self.linucb_agent=self.linucb_agent_backup
            self.linucb_agent_backup=newb
            self.bandit_swap_e=self.e
            print('we swap bandit here')

        self.buffer_count=0


    def measure_rnn(self,state,j,tick):
        if j > 1 + config.TRAIN_CONFIG['frame_skip']:
            temp_init = self.init_state_list[-config.TRAIN_CONFIG['frame_skip']]
            initial_rnn_cstate = temp_init[0]
            initial_rnn_hstate = temp_init[1]
        else:
            initial_rnn_cstate = np.zeros((1, 1, config.TRAIN_CONFIG['lstm_unit']))
            initial_rnn_hstate = np.zeros((1, 1, config.TRAIN_CONFIG['lstm_unit']))
        rnn_value, initial_rnn_state = self.sess.run([self.agent.main_rnn_value, self.agent.rnn_out_state],
                                                         feed_dict={self.agent.training_phase: 0,
                                                                    self.agent.scalarInput: state,
                                                                    self.agent.rnn_cstate_holder: initial_rnn_cstate,
                                                                    self.agent.rnn_hstate_holder: initial_rnn_hstate,
                                                                    self.agent.iter_holder: [np.array([tick])],
                                                                    self.agent.eps_holder: [np.array([self.e])],
                                                                    self.agent.trainLength: 1,
                                                                    self.agent.batch_size: 1})
        return rnn_value,initial_rnn_state


    def take_action(self,state,feature,j,tick):
        a = [st for st in range(self.N_station)]

        # predict_score = sess.run(linear_model.linear_Yh, feed_dict={linear_model.linear_X: [feature]})
        predict_score = self.linucb_agent.return_upper_bound(feature)
        predict_score = predict_score/self.distance
        valid = predict_score > config.TRAIN_CONFIG['elimination_threshold']
        invalid = predict_score <= config.TRAIN_CONFIG['elimination_threshold']
        rnn_value,initial_rnn_state=self.measure_rnn(state,j,tick)
        self.init_state_list.append(initial_rnn_state)


        self.init_state_list.append(initial_rnn_state) #record rnn state
        for station in range(self.N_station):
            if self.env.taxi_in_q[station]:
                rand_num = np.random.rand(1)
                a1 = self.agent.predict(rnn_value, predict_score[station,:].copy(), self.e, station, rand_num,
                                   valid[station,:], invalid[station,:])
                a[station] = a1  # action performed by DRQN
            else:
                a[station] = self.N_station  # no available vehicles

        return a

    def buffer_record(self, state):
        self.global_epi_buffer.append(state)
        self.global_bandit_buffer.append(state)
        self.buffer_count += 1
        if self.buffer_count >= (1 + config.TRAIN_CONFIG['frame_skip']) * self.trace_length:
            for it in range(self.trace_length):
                bufferArray = np.array(self.global_epi_buffer)
                self.exp_replay.add(bufferArray[it:it + config.TRAIN_CONFIG['frame_skip'] * self.trace_length:config.TRAIN_CONFIG[
                    'frame_skip']])  # skippig 3 frames in the replay buffer
            self.global_epi_buffer = self.global_epi_buffer[self.trace_length:]
            self.buffer_count -= self.trace_length


    def epsilon_decay(self):
        self.total_steps += 1
        if self.total_steps > self.pre_train_steps:
            # start training here
            if self.e > self.endE:
                self.e -= self.stepDrop

    def update_bandit(self):
        #update bandit every 500 steps
        if self.total_steps%(30)==0:
            linubc_train = self.bandit_buffer.sample(self.batch_size * 2)
            if self.total_steps<=2*self.pre_train_steps:
                self.linucb_agent.update(linubc_train[:, 4], linubc_train[:, 1], linubc_train[:, 5])
                self.linucb_agent_backup.update(linubc_train[:, 4], linubc_train[:, 1], linubc_train[:, 5])
            else:
                self.linucb_agent_backup.update(linubc_train[:, 4], linubc_train[:, 1], linubc_train[:, 5])

    def train_agent(self):
        # use a single buffer
        if self.total_steps > self.pre_train_steps:
            # train linear multi-arm bandit first, we periodically update this (every 10*update_fequency steps)
            if self.total_steps % (self.update_freq) == 0:
                self.total_train_iter+=1
                #update target network
                self.agent.update_target_net()
                for station in range(self.N_station):
                    # visual and normalization
                    trainBatch = self.exp_replay.sample(self.batch_size, self.trace_length)

                    #very important: convert the structure of the batch
                    trainBatch=self.convert_batch_time(trainBatch,self.batch_size,self.trace_length)
                    #very important
                    past_train_eps = np.vstack(trainBatch[:, 7])
                    past_train_iter = np.vstack(trainBatch[:, 8])
                    tr, t_action = self.agent.train_prepare(trainBatch, station)
                    tr[t_action==self.N_station]=0 #no reward for those actions
                    target_in = np.vstack(trainBatch[:, 3])
                    train_in = np.vstack(trainBatch[:, 0])
                    train_predict_score = self.linucb_agent.return_upper_bound_batch(np.vstack(trainBatch[:, 6]))/self.distance[station,:]
                    tp = train_predict_score.copy()
                    invalid = tp <= self.e_threshold
                    valid = tp > self.e_threshold
                    tp[invalid] = 1e4
                    tp[valid] = 0
                    tp[:,station]=0
                    tp_new = 1e4 * np.ones((self.batch_size * self.trace_length, self.N_station + 1))
                    tp_new[:, :-1] = tp
                    tp = tp_new
                    station_in=[station]*self.batch_size*self.trace_length
                    tz = self.sess.run(self.agent.targetZ[station],
                                  feed_dict={self.agent.training_phase: 0, self.agent.scalarInput: target_in,
                                             self.agent.iter_holder: past_train_iter, self.agent.eps_holder: past_train_eps,
                                             self.agent.predict_score[station]: tp, self.agent.rewards[station]: tr,
                                             self.agent.trainLength: self.trace_length, self.agent.batch_size:self.batch_size,self.agent.station_id:station_in})

                    self.sess.run(self.agent.updateModel[station],
                             feed_dict={self.agent.training_phase: 1, self.agent.targetQ[station]: tz, self.agent.rewards[station]: tr,
                                        self.agent.actions[station]: t_action, self.agent.scalarInput: train_in,
                                        self.agent.iter_holder: past_train_iter, self.agent.eps_holder: past_train_eps,
                                        self.agent.trainLength: self.trace_length, self.agent.batch_size: self.batch_size,self.agent.station_id:station_in})
                    # print('training loss is:....',loss)


    def process_bandit_buffer(self):
        future_steps=self.trace_length
        tmask = np.linspace(0, 1, num=future_steps + 1)
        quantile_mask=tmask
        #pdeta=0.5;
        #quantile_mask=scipy.stats.norm.cdf(scipy.stats.norm.ppf(tmask)-pdeta)
        quantile_mask = np.diff(quantile_mask) # rescale the distribution to favor risk neutral or risk-averse behavior

        for epi in range(len(self.global_bandit_buffer)-future_steps-1):
            score=np.array([self.global_bandit_buffer[epi+k][0][5] for k in range(future_steps)]).T.dot(quantile_mask)
            record=self.global_bandit_buffer[epi]
            record[0][5]=score; #replay the score
            self.bandit_buffer.add(record)

    def convert_batch_time(self,batch, batch_size, trace_length):
        # batch is original in time x batch format, now we convert it into batch x time format
        new_batch = []
        for i in range(trace_length):
            for j in range(batch_size):
                new_batch.append(batch[j * trace_length + i])
        return (np.array(new_batch))

    def load_params(self):
        with open('simulation_input.dat', 'rb') as fp:
            simulation_input = pickle.load(fp)
            self.N_station = simulation_input['N_station']
            self.OD_mat = simulation_input['OD_mat']
            self.distance = simulation_input['distance']
            self.travel_time = simulation_input['travel_time']
            self.arrival_rate = simulation_input['arrival_rate']
            self.taxi_input = simulation_input['taxi_input']
            self. exp_dist = simulation_input['exp_dist']
            # Setting the training parameters
            self.batch_size = config.TRAIN_CONFIG['batch_size']
            self.trace_length = config.TRAIN_CONFIG['trace_length']  # How long each experience trace will be when training
            self.update_freq = config.TRAIN_CONFIG['update_freq']  # How often to perform a training step.
            self. lstm_units = config.TRAIN_CONFIG['lstm_unit']
            self. e_threshold = config.TRAIN_CONFIG['elimination_threshold']
            self. y = config.TRAIN_CONFIG['y']  # Discount factor on the target Q-values
            self. startE = config.TRAIN_CONFIG['startE']  # Starting chance of random action
            self. endE = config.TRAIN_CONFIG['endE']  # Final chance of random action
            self. anneling_steps = config.TRAIN_CONFIG['anneling_steps']  # How many steps of training to reduce startE to endE.
            self. num_episodes = config.TRAIN_CONFIG[
                'num_episodes']  # How many episodes of game environment to train network with.
            self.load_model = config.TRAIN_CONFIG['load_model']  # Whether to load a saved model.
            self. warmup_time = config.TRAIN_CONFIG['warmup_time'];
            self. path = "./small_network_save_model"  # The path to save our model to.
            self.h_size = config.TRAIN_CONFIG['h_size']
            self. max_epLength = config.TRAIN_CONFIG['max_epLength']
            self.pre_train_steps =self. max_epLength * 1  # How many steps of random actions before training begins.
            self.softmax_action = config.TRAIN_CONFIG['softmax_action']
            self.silent = config.TRAIN_CONFIG['silent']  # do not print training time
            self.prioritized = config.TRAIN_CONFIG['prioritized']
            self.rng_seed = config.TRAIN_CONFIG['random_seed']
            self.use_gpu=1


