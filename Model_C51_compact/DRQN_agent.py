#Xinwu Qian 2019-02-07

#Agent file for agents who follow DRQN to update their rewards

import os
import numpy as np
import tensorflow as tf
import network
import time
import tensorflow.contrib.slim as slim
import config
from scipy.stats import norm

import tf_conv_net



#lets define a memory efficient drqn_agent()

class drqn_agent_efficient():
    def __init__(self,N_station,h_size,lstm_units,tau,sess,batch_size,train_length,is_gpu=0,ckpt_path=None):
        self.N_station=N_station;
        self.first_action=3; #relocate or not or no available actions because of relocation
        self.h_size=h_size;
        self.lstm_units=lstm_units;
        self.tau=tau;
        self.sess=sess;
        self.train_length=train_length;
        self.use_gpu=is_gpu;
        self.ckpt_path=ckpt_path;


        self.count_single_act=0
        self.num_gpu=2
        #QR params
        self.N=21; #number of quantiles
        self.n_head=5; #number of bootstrap head
        self.k=1; #huber loss
        self.gamma=config.TRAIN_CONFIG['y']
        self.conf=1


        #risk averse
        # risk seeking behavior
        tmask = np.linspace(0, 1, num=self.N + 1)
        self.eta = config.NET_CONFIG['eta']
        # self.quantile_mask = tmask ** self.eta / (tmask ** self.eta + (1 - tmask) ** self.eta) ** (
        #             1 / self.eta)
        if config.NET_CONFIG['Risk_Distort']:
            self.quantile_mask=norm.cdf(norm.ppf(tmask)-self.eta)
        else:
            self.quantile_mask=tmask

        self.quantile_mask = np.diff(self.quantile_mask) # rescale the distribution to favor risk neutral or risk-averse behavior


        #risk seeking
        # self.mask=np.concatenate([np.zeros(self.N-self.N//2),np.ones(self.N//2)])
        # self.quantile_mask=self.mask

        #place holders.
        with tf.device('/cpu:0'):
            self.scalarInput = [tf.placeholder(shape=[None, N_station * N_station * 6], dtype=tf.float32, name='main_input') for i in range(self.N_station)]
            self.target_eval_scalarInput = [
                tf.placeholder(shape=[None, N_station * N_station * 6], dtype=tf.float32, name='main_input') for i in
                range(self.N_station)]
            self.trainLength = tf.placeholder(dtype=tf.int32, name='trainlength')
            self.batch_size = tf.placeholder(dtype=tf.int32, name='batchsize')
            iter_holder = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='iterholder')
            eps_holder = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='epsholder')
            self.training_phase = tf.placeholder(tf.bool, name='istraining')
            self.relocation_decision=[tf.placeholder(shape=[None,N_station*self.first_action],dtype=tf.float32,name='relocation_decision') for i in range(self.N_station)]
            self.targetQ = []
            self.actions = []
            self.rewards = []
            self.station_score = []
            self.station_relo_score=[]
            self.predict_score = []
            self.predict_relo_score=[]
            self.iter_holder=[]
            self.eps_holder=[]
            self.station_id=[]
            self.vehicle_available=[tf.placeholder(tf.float32,shape=[None,self.first_action]) for i in range(self.N_station)]
            bs_mask=[tf.placeholder(tf.float32,shape=[None,1]) for _ in range(self.N_station)] #mask for gradient propogation
            self.bootstrap_mask=[bs_mask for _ in range(self.n_head)]

            self.rnn_holder = tf.placeholder(shape=[None, self.lstm_units], dtype=tf.float32, name='main_input')
            self.rnn_cstate_holder = tf.placeholder(shape=[None, None, self.lstm_units], dtype=tf.float32, name='main_input')
            self.rnn_hstate_holder = tf.placeholder(shape=[None, None, self.lstm_units], dtype=tf.float32, name='main_input')

            station_id = tf.placeholder(tf.int32, shape=[None], name='station_id')
            t_targetQ=[]
            for i in range(N_station):
                targetQ = tf.placeholder(shape=[None, self.N], dtype=tf.float32)
                actions = tf.placeholder(shape=[None], dtype=tf.int32)
                rewards = tf.placeholder(shape=[None], dtype=tf.float32)
                predict_score = tf.placeholder(dtype=tf.float32, shape=[None, self.N_station+1])
                t_targetQ.append(targetQ)
                self.actions.append(actions)
                self.rewards.append(rewards)
                self.predict_score.append(predict_score)
                self.predict_relo_score.append(tf.placeholder(dtype=tf.float32, shape=[None, self.first_action]))
                self.station_id.append(station_id)
                self.iter_holder.append(iter_holder)
                self.eps_holder.append(eps_holder)
            self.targetQ=[t_targetQ for _ in range(self.n_head)] #h by N_station
            self.Adv_fun_train=[]
            self.Adv_fun_predict=[]
            self.Adv_target=[]
            self.relo_predict_score=np.array([0,0,1e5])
            self.relo_globalnorm=[]
            self.act_globalnorm=[]

            # ops.
            self.mainQout = [[] for i in range(self.n_head)]
            self.mainQout_first=[[] for i in range(self.n_head)]
            self.targetQout = [[] for i in range(self.n_head)]
            self.targetQout_first=[[] for i in range(self.n_head)]
            self.mainPredict = [[] for i in range(self.n_head)]
            self.mainPredict_first=[[] for i in range(self.n_head)]
            self.mainpredict_relo2=[[] for i in range(self.n_head)]
            self.updateModel = [[] for i in range(self.n_head)]
            self.updateModel_first=[[] for i in range(self.n_head)]
            self.targetZ = [[] for i in range(self.n_head)]
            self.targetZ_first=[[] for i in range(self.n_head)]
            self.Qout2=[[] for i in range(self.n_head)]
            self.Qout2_first=[[] for i in range(self.n_head)]

            self.V_relo_target=[[] for i in range(self.n_head)]
            self.V_relo=[[] for i in range(self.n_head)]
            self.V2_relo=[[] for i in range(self.n_head)]

            self.predict_all = [[] for i in range(self.n_head)]

            self.predict_relo_mean=[[] for i in range(self.N_station)]
            self.predict_act_mean=[[] for i in range(self.n_head)]
            self.predict_relo_var=[[] for i in range(self.N_station)]
            self.predict_act_var=[[] for i in range(self.n_head)]

            self.predict_relo_all = [[] for i in range(self.n_head)]
            self.predict_relo_all2 = [[] for i in range(self.n_head)]
            self.mainQout_first_all = [[] for i in range(self.n_head)]
            self.mainQout_all = [[] for i in range(self.n_head)]
            self.targetQout_first_all=[[] for i in range(self.n_head)]
            self.targetQout_all=[[] for i in range(self.n_head)]
            self.targetZ_all=[[] for i in range(self.n_head)]
            self.targetZ_first_all=[[] for i in range(self.n_head)]



    #first level to determine if relocate or not
    def init_main_first(self,reuse=None):
        with tf.device('/gpu:0'):
            myScope_main = 'main'
            lstm=self.build_lstm(myScope_main+'_lstm')
            self.main_lstm=lstm
            imageIn = self.scalarInput[0]
            conv_flat = self.convolution(imageIn, myScope=myScope_main, reuse=None)
            rnn, _= self.get_rnn(conv_flat,lstm,training=True)

            #for output rnn values during prediction
            rnn_out,rnn_state_out=self.get_rnn(conv_flat,lstm,training=False)
            self.main_rnn_value_first=rnn_out
            self.rnn_out_state_first=rnn_state_out
        #head and stations
        for i in range(self.N_station):
            with tf.device('/gpu:'+str(i%self.num_gpu)):
                for h in range(self.n_head):
                    Qout=self.predict_head_Qout(rnn,station=i,head=h,myScope=myScope_main,reuse=None)

    def init_target_first(self):
        with tf.device('/gpu:0'):
            myScope_main = 'target'
            lstm=self.build_lstm(myScope_main+'_lstm')
            self.target_lstm=lstm
            imageIn = self.scalarInput[0]
            conv_flat = self.convolution(imageIn, myScope=myScope_main, reuse=None)
            rnn, rnn_state = self.get_rnn(conv_flat,lstm)
        for i in range(self.N_station):
            with tf.device('/gpu:'+str(i%self.num_gpu)):
                for h in range(self.n_head):
                    Qout=self.predict_head_Qout(rnn,station=i,head=h,myScope=myScope_main,reuse=None)

    def convolution(self,image_in,myScope,reuse=None):
        imageIn = tf.reshape(image_in, shape=[-1, self.N_station, self.N_station, 6])
        input_conv = tf.pad(imageIn, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT")  # reflect padding!
        conv = tf_conv_net.build_convolution(myScope, input_conv, config.NET_CONFIG['case'], self.training_phase,reuse)
        convFlat = tf.reshape(slim.flatten(conv), [self.trainLength, self.batch_size, self.h_size])
        return convFlat

    def build_lstm(self,scope):
        if self.use_gpu:
            print('Using CudnnLSTM')
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.lstm_units,
                                                  name=scope)

        else:
            print('Using LSTMfused')
            lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.lstm_units, name=scope)
        return lstm

    def get_rnn(self,conv,lstm,training=True):
        if training:
            rnn, rnn_state = lstm(inputs=conv, training=training)
        else:
            my_initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.rnn_cstate_holder, self.rnn_hstate_holder)
            rnn,rnn_state=lstm(inputs=conv, initial_state=my_initial_state, training=False)
        rnn = tf.reshape(rnn, shape=[-1, self.lstm_units])
        return rnn,rnn_state

    def predict_head_Qout(self,rnn,station,head,myScope,reuse=None):
        ## Input rnn: [None, lstm_units]
        ## Ouput: Q value [None, N_station, N]
        ## args: station--station id, head -- head id, myScope: distinguish target or main
        streamA, streamV = tf.split(rnn, 2, 1)
        streamV_local = streamV  # tf.concat([streamV,tf.one_hot(self.station_id[station], self.N_station, dtype=tf.float32)], -1)
        streamA_local = streamA  # streamA #tf.concat([streamA,tf.one_hot(self.station_id[station],self.N_station,dtype=tf.float32)],-1)
        Qout = self.build_head(rnn, n_actions=self.N_station+1, head_id=head, station_id=station,Scope=myScope, reuse=reuse)
        return Qout

        # bootstrap
    def build_head(self, rnn, Scope, n_actions, head_id, station_id,reuse=None):
        head_name=Scope+str(head_id)+str(station_id)

        streamV = tf.layers.dense(rnn, self.lstm_units//2, name=head_name + 'VW1',bias_initializer=tf.contrib.layers.xavier_initializer(), activation='relu', reuse=reuse)  # advantage
        Value = tf.layers.dense(streamV, self.N, name=head_name + 'VW2',bias_initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse)  # advantage
        Value = tf.reshape(Value, [-1, 1, self.N])
        
        streamA = tf.layers.dense(rnn, self.lstm_units//2, name=head_name + 'AW1',bias_initializer=tf.contrib.layers.xavier_initializer(), activation='relu', reuse=reuse)  # advantage
        localA = tf.layers.dense(streamA, n_actions* self.N, name=head_name+ 'AW2', bias_initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse)  # advantage
        
       #convertA = tf.reshape(localA, [-1, n_actions, self.N])
        maxA = tf.reduce_mean(localA, -1, keepdims=True)
        Qout = Value+tf.reshape(tf.subtract(localA, maxA,name=Scope + str(head_id) + '_unshaped_Qout'),[-1,n_actions,self.N])
        #Qout = tf.reshape(Qt, [-1, n_actions, self.N])  # reshape it to N_station by self.atoms dimension
        return Qout

    def build_actions(self,coeff=1,epsilon=1e-5):
        #build action selection which returns the minimum information gain
        relo_regret_list = []
        act_regret_list=[]
        with tf.device('/cpu:0'):
            self.head_Qout=[[] for _ in range(self.n_head)]
            for head in range(self.n_head):
                for station in range(self.N_station):
                    Qout = self.predict_head_Qout(self.rnn_holder, station=station, head=head, myScope='main', reuse=True)
                    self.head_Qout[head].append(tf.sort(Qout,-1))


    def update_target_net(self):
        network.updateTarget(self.targetOps, self.sess)


    def return_Q(self,scope,input,station,head):
        if scope=='main':
            lstm=self.main_lstm
        elif scope=='target':
            lstm=self.target_lstm
        else:
            print('Error: please provide scope name specifying using main or target network')
        conv_flat = self.convolution(input, myScope=scope, reuse=True)
        rnn, _ = self.get_rnn(conv_flat,lstm,training=True)
        Qout=self.predict_head_Qout(rnn,station=station,head=head,myScope=scope,reuse=True)
        return Qout

    def build_train(self):
        #we implement global learning rate decay!
        gradient=[]
        variable=[]
        gclip=10; #gradient clipping value
        # lossmask1 = tf.zeros([self.trainLength // 2, self.batch_size])
        # lossmask2 = tf.ones([self.trainLength // 2, self.batch_size])
        # lossmask = tf.concat([lossmask1, lossmask2], 0)
        # lossmask = tf.reshape(lossmask, [-1])
        #trainer = tf.train.AdamOptimizer(epsilon=0.001,learning_rate=config.TRAIN_CONFIG['learning_rate_opt'], name='Adam_opt_act')
        trainer=tf.train.AdamOptimizer(config.TRAIN_CONFIG['learning_rate_opt'])
        for i in range(self.N_station):
            with tf.device('/gpu:'+str(i%self.num_gpu)): #place on each gpu separately
                for h in range(self.n_head):
            # for action taking,,,,,,,,,,,,,,,,,,,,,,,,,,,,
                    mainQ=self.return_Q('main', self.target_eval_scalarInput[i], i, h)
                    targetQ=self.return_Q('target', self.target_eval_scalarInput[i], i, h)
                    q = tf.reduce_mean(mainQ, axis=-1)
                    main_q = tf.subtract(q, self.predict_score[i])
                    main_act = tf.argmax(main_q, axis=-1)
                    target_mask = tf.one_hot(main_act, self.N_station+1, dtype=tf.float32)  # out: [None, n_actions]
                    target_mask = tf.expand_dims(target_mask, axis=-1)  # out: [None, n_actions, 1]
                    #for action network, the target network is the aciton taking network!
                    selected_target= tf.reduce_sum(targetQ * target_mask, axis=1)  # out: [None, N]
                    rew_t = tf.expand_dims(self.rewards[i], axis=-1)
                    target_z = rew_t + self.gamma * tf.stop_gradient(selected_target) #make sure you are not updating target network as well

                    mainQ_train=self.return_Q('main', self.scalarInput[i], i, h)
                    mainz=self._compute_estimate(mainQ_train,self.actions[i],self.N_station+1)
                    loss = self._compute_loss(mainz,target_z)
                    loss = tf.reduce_mean(loss*self.bootstrap_mask[h][i]) #minibatch average
                    # In order to only propogate accurate gradients through the network, we will mask the first
                    # half of the losses for each trace as per Lample & Chatlot 2016
                    gradients, variables = zip(*trainer.compute_gradients(loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, gclip)
                    gbn=tf.global_norm(gradients)
                    self.act_globalnorm.append(gbn)
                    gradient+=gradients
                    variable+=variables

        with tf.device('/cpu:0'):
            updateModel = trainer.apply_gradients(zip(gradient, variable))
            self.updateModel=updateModel

    def drqn_build(self):
        self.init_main_first()
        self.init_target_first()
        self.build_train()
        self.build_actions()
        # self.main_trainables = tf.trainable_variables(scope='DRQN_main_')
        self.trainables = tf.trainable_variables()
        # self.target_trainables = tf.trainable_variables(scope='DRQN_target')

        # store the name and initial values for target network
        self.targetOps = network.updateTargetGraph(self.trainables, self.tau)
        # self.update_target_net()

        print("Agent network initialization complete with:",str(self.N_station),' agents')



    def predict(self, rnn,predict_score,e,station,rng,valid,invalid,relo_action,tick,Q):
        # make the prediction
       # print(self.conf)
        #
        valid[station]=True
        if e==1:
            action=np.random.randint(self.N_station)
        # if np.random.random()>e:
        else:
            localQ=Q[station][0] #format N_station x N
            median=np.median(localQ,axis=1) 
            mean=np.mean(localQ,axis=1) 
            upper_var=np.sqrt(np.mean((localQ[:,self.N//2+1:]-np.expand_dims(median,axis=1))**2,axis=1))
            ucb_bound=mean+50*(e**2)*upper_var
            v=-1e8
            act=0
            for i in range(self.N_station):
                if ucb_bound[i]>v:
                    if valid[i]==True:
                        v=ucb_bound[i]
                        act=i
            action=act
        return action



    def train_prepare(self, trainBatch,station):

        #params:
        # Q1 and Q2 in the shape of [batch*length, N_station, N]
        reward=np.array([r[station] for r in trainBatch[:,2]])
        action=np.array([a[station] for a in trainBatch[:,1]])
        #reward[action==self.N_station]=0;
        return reward,action


    def _compute_estimate(self, agent_net,action,n_actions):
        """Select the return distribution Z of the selected action
        Args:
          agent_net: `tf.Tensor`, shape `[None, n_actions, N]. The tensor output from `self._nn_model()`
            for the agent
            action: 'tf.Tensor', shape '[None]
        Returns:
          `tf.Tensor` of shape `[None, N]`
        """
        a_mask = tf.one_hot(action, n_actions, dtype=tf.float32)  # out: [None, n_actions]
        a_mask = tf.expand_dims(a_mask, axis=-1)  # out: [None, n_actions, 1]
        z = tf.reduce_sum(agent_net * a_mask, axis=1)  # out: [None, N]
        return z


    def _select_target(self, main_out,target_out,predict_score):
        """Select the QRDQN target distributions - use the greedy action from E[Z]
        Args:
        main_out: `tf.Tensor`, shape `[None, n_actions, N]. The tensor output from `self._nn_model()`
            for the main network

          target_out: `tf.Tensor`, shape `[None, n_actions, N]. The tensor output from `self._nn_model()`
            for the target network
        Returns:
          `tf.Tensor` of shape `[None, N]`
        """

        #Choose the action from main network
        main_z=main_out
        main_q=tf.reduce_mean(main_z,axis=-1)
        main_q=tf.subtract(main_q,predict_score)
        main_act=tf.argmax(main_q,axis=-1)

        #Return the evaluation from target network

        target_mask = tf.one_hot(main_act, self.N_station, dtype=tf.float32)  # out: [None, n_actions]
        target_mask = tf.expand_dims(target_mask, axis=-1)  # out: [None, n_actions, 1]
        target_z = tf.reduce_sum(target_out * target_mask, axis=1)  # out: [None, N]
        return target_z


    def _compute_backup(self, target,reward):
        """Compute the QRDQN backup distributions
        Args:
          target: `tf.Tensor`, shape `[None, N]. The output from `self._select_target()`
        Returns:
          `tf.Tensor` of shape `[None, N]`
        """
        # Compute the projected quantiles; output shape [None, N]
        rew_t = tf.expand_dims(reward, axis=-1)
        target_z = rew_t + self.gamma * target
        return target_z

    def _compute_loss(self, mainQ, targetQ):
        """Compute the QRDQN loss.
        Args:
          mainQ: `tf.Tensor`, shape `[None, N]. The tensor output from `self._compute_estimate()`
          targetQ: `tf.Tensor`, shape `[None, N]. The tensor output from `self._compute_backup()`
        Returns:
          `tf.Tensor` of scalar shape `()`
        """

        # Compute the tensor of mid-quantiles
        mid_quantiles = (np.arange(0, self.N, 1, dtype=np.float64) + 0.5) / float(self.N)
        mid_quantiles = np.asarray(mid_quantiles, dtype=np.float32)
        mid_quantiles = tf.constant(mid_quantiles[None, None, :], dtype=tf.float32)

        # Operate over last dimensions to average over samples (target locations)
        td_z = tf.expand_dims(targetQ, axis=-2) - tf.expand_dims(mainQ, axis=-1)
        # td_z[0] =
        # [ [tz1-z1, tz2-z1, ..., tzN-z1],
        #   [tz1-z2, tz2-z2, ..., tzN-z2],
        #   ...
        #   [tz1-zN, tzN-zN, ..., tzN-zN]  ]
        indicator_fn = tf.to_float(td_z < 0.0)  # out: [None, N, N]

        # Compute the quantile penalty weights
        quant_weight = mid_quantiles - indicator_fn  # out: [None, N, N]
        # Make sure no gradient flows through the indicator function. The penalty is only a scaling factor
        quant_weight = tf.stop_gradient(quant_weight)

        # Pure Quantile Regression Loss
        if self.k == 0:
            quantile_loss = quant_weight * td_z  # out: [None, N, N]
        # Quantile Huber Loss
        else:
            quant_weight = tf.abs(quant_weight)
            be=tf.abs(td_z)
            huber_loss = tf.where(be<self.k,0.5*tf.square(be),self.k*(be-0.5*self.k))
            quantile_loss = quant_weight * huber_loss  # out: [None, N, N]

        quantile_loss = tf.reduce_mean(quantile_loss, axis=-1)  # Expected loss for each quantile
        loss = tf.reduce_sum(quantile_loss, axis=-1)  # Sum loss over all quantiles

        #hysteria loss
        #loss_hyst=tf.where(tf.reduce_mean(targetQ,axis=-1)<tf.reduce_mean(mainQ,axis=-1),0.3*loss,loss)
        # loss = tf.reduce_mean(loss)  # Average loss over the batch

        return loss
