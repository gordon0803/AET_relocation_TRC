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

class drqn_agent_action():
    def __init__(self,N_station,h_size,lstm_units,tau,sess,batch_size,train_length,is_gpu=0,ckpt_path=None):
        self.N_station=N_station;
        self.N_actions=2; #relocate or not
        self.h_size=h_size;
        self.lstm_units=lstm_units;
        self.tau=tau;
        self.sess=sess;
        self.train_length=train_length;
        self.use_gpu=is_gpu;
        self.ckpt_path=ckpt_path;

        #QR params
        self.N=51; #number of quantiles
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
        self.scalarInput = tf.placeholder(shape=[None, N_station * N_station * 5], dtype=tf.float32, name='main_input')
        self.trainLength = tf.placeholder(dtype=tf.int32, name='trainlength')
        self.batch_size = tf.placeholder(dtype=tf.int32, name='batchsize')
        self.iter_holder = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='iterholder')
        self.eps_holder = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='epsholder')
        self.training_phase = tf.placeholder(tf.bool, name='istraining')
        self.station_id=tf.placeholder(tf.int32,shape=[None],name='station_id')

        self.targetQ = []
        self.actions = []
        self.rewards = []
        self.main_rnn_value = []
        self.station_score = []
        self.predict_score = []
        self.rnn_holder = tf.placeholder(shape=[None, self.lstm_units], dtype=tf.float32, name='main_input')
        self.rnn_cstate_holder = tf.placeholder(shape=[1, None, self.lstm_units], dtype=tf.float32, name='main_input')
        self.rnn_hstate_holder = tf.placeholder(shape=[1, None, self.lstm_units], dtype=tf.float32, name='main_input')

        for i in range(N_station):
            targetQ = tf.placeholder(shape=[None, self.N], dtype=tf.float32)
            actions = tf.placeholder(shape=[None], dtype=tf.int32)
            rewards = tf.placeholder(shape=[None], dtype=tf.float32)
            predict_score = tf.placeholder(dtype=tf.float32, shape=[None, self.N_station + 1])
            self.targetQ.append(targetQ)
            self.actions.append(actions)
            self.rewards.append(rewards)
            self.predict_score.append(predict_score)

        self.Adv_fun_train=[]
        self.Adv_fun_predict=[]
        self.Adv_target=[]
        # nets
        # self.conv1=[]
        # self.conv2=[]
        # self.conv3=[]
        # self.conv4=[]
        # self.rnn=[]

        # ops.
        self.mainQout = []
        self.targetQout = []
        self.mainPredict = []
        self.updateModel = []
        self.targetZ = []
        self.Qout2=[]



    def build_main(self):
        myScope_main = 'DRQN_Main_'
        imageIn = tf.reshape(self.scalarInput, shape=[-1, self.N_station, self.N_station, 5], name=myScope_main + 'in1')
        input_conv = tf.pad(imageIn, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT",
                            name=myScope_main + 'in2')  # reflect padding!

        #bn = tf.layers.batch_normalization(conv3, training=self.training_phase,trainable=True)
        conv=tf_conv_net.build_convolution(myScope_main,input_conv,config.NET_CONFIG['case'],self.training_phase)

        if self.use_gpu:
            print('Using CudnnLSTM')
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.lstm_units, name=myScope_main + '_lstm')

        else:
            print('Using LSTMfused')
            lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.lstm_units, name=myScope_main + '_lstm')

        convFlat = tf.reshape(slim.flatten(conv), [self.trainLength,self.batch_size, self.h_size],
                              name=myScope_main + '_convlution_flattern')
        #
        iter=tf.reshape(self.iter_holder,[self.trainLength,self.batch_size,1])
        eps=tf.reshape(self.eps_holder,[self.trainLength,self.batch_size,1])
        convFlat=tf.concat([convFlat,iter,eps],axis=-1)

        rnn, rnn_state = lstm(inputs=convFlat,training=True)
        rnn = tf.reshape(rnn, shape=[-1, self.lstm_units], name=myScope_main + '_reshapeRNN_out')

        my_initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.rnn_cstate_holder,self.rnn_hstate_holder)
        rnnin,rnn_out_state=lstm(inputs=convFlat,initial_state=my_initial_state,training=False)
        self.main_rnn_value.append(rnnin)
        self.rnn_out_state=rnn_out_state
        streamA, streamV = tf.split(rnn, 2, 1, name=myScope_main + '_split_streamAV')

        streamA2, streamV2 = tf.split(self.rnn_holder, 2, 1, name=myScope_main + '_split_streamAV')
        #Vg, Vl = tf.split(streamV, 2, 1, name=myScope_main + '_split_streamVgl_train')
        #Vg2, Vl2 = tf.split(streamV2, 2, 1, name=myScope_main + '_split_streamVgl_predict')



        V_list=[]
        V_predict=[]
        A_list=[]
        for i in range(self.N_station):
            myScope = 'DRQN_main_' + str(i)
            #localValue = tf.layers.dense(Vl, 1, name=myScope + 'VW', activation='linear', reuse=None)  # advantage
            #localValue2 = tf.layers.dense(Vl2, 1, name=myScope + 'VW', activation='linear',
             #                        reuse=True)  # advantage
            streamA_local=tf.concat([streamA,tf.one_hot(self.station_id,self.N_station,dtype=tf.float32)],-1)
            streamA2_local=tf.concat([streamA2,tf.one_hot(self.station_id,self.N_station,dtype=tf.float32)],-1)
            streamV_local=tf.concat([streamV, tf.one_hot(self.station_id, self.N_station, dtype=tf.float32)], -1)
            streamV2_local=tf.concat([streamV2, tf.one_hot(self.station_id, self.N_station, dtype=tf.float32)], -1)
            if i==0:
                localA = tf.layers.dense(streamA_local, (self.N_station + 1) * self.N, name=myScope_main + 'AW', activation='linear', reuse=None)  # advantage
                Value = tf.layers.dense(streamV_local, 1, name=myScope_main + 'VW', activation='linear',
                                        reuse=None)  # advantage
                Value2 = tf.layers.dense(streamV2_local, 1, name=myScope_main + 'VW', activation='linear',
                                         reuse=True)  # advantage
            else:
                localA = tf.layers.dense(streamA_local, (self.N_station + 1) * self.N, name=myScope_main + 'AW',
                                         activation='linear', reuse=True)  # advantage
                Value = tf.layers.dense(streamV_local, 1, name=myScope_main + 'VW', activation='linear',
                                        reuse=True)  # advantage
                Value2 = tf.layers.dense(streamV2_local, 1, name=myScope_main + 'VW', activation='linear',
                                         reuse=True)  # advantage


            localA2 = tf.layers.dense(streamA2_local, (self.N_station + 1) * self.N, name=myScope_main + 'AW', activation='linear',reuse=True)  # advantage

            # localA=tf.reshape(localA,[-1,self.N_station+1,self.N])
            # Value = tf.reshape(tf.tile(Value, [1, self.N_station + 1]),
            #                    [self.batch_size * self.trainLength, self.N_station + 1, self.N])
            V_list.append(Value)
            V_predict.append(Value2)
            A_list.append(localA)
            # localA2 = tf.reshape(localA2, [-1, self.N_station + 1, self.N])

            # Value2 = tf.reshape(tf.tile(Value2, [1, self.N_station + 1]),
            #                    [self.batch_size * self.trainLength, self.N_station + 1, self.N])

            Qt2 = Value2 + tf.subtract(localA2, tf.reduce_mean(localA2, axis=1, keepdims=True),
                                       name=myScope + '_unshaped_Qout')

            Qout2 = tf.reshape(Qt2, [-1, self.N_station+1, self.N])  # reshape it to N_station by self.atoms dimension
            q = tf.reduce_mean(tf.sort(Qout2, axis=-1) * self.quantile_mask, axis=-1)

            station_vec = tf.concat([tf.ones(i),tf.ones(1),tf.ones(self.N_station-i)], axis=0)
            station_score = tf.multiply(self.predict_score[i], station_vec)  # mark self as 0
            self.station_score.append(station_score)
            # predict based on the 95% confidence interval
            # predict = tf.argmax(tf.subtract(tf.reduce_sum(mean,tf.scalar_mul(self.conf,std)),self.station_score[i]), 1, name=myScope + '_prediction')
            predict = tf.argmax(tf.subtract(q, self.station_score[i]), 1, name=myScope + '_prediction')
            self.mainPredict.append(predict)

        self.V_list=V_list
        sumV=tf.reduce_sum(V_list,axis=0) #sum of valeus
        for i in range(self.N_station):
            myScope = 'DRQN_main_' + str(i)
            Qt =V_list[i]+tf.subtract(A_list[i], tf.reduce_mean(A_list[i], axis=1, keepdims=True),
                                     name=myScope + '_unshaped_Qout')
            Qout = tf.reshape(Qt, [-1, self.N_station+1, self.N])  # reshape it to N_station by self.atoms dimension
            self.mainQout.append(Qout)






    def build_target(self):
        myScope_main = 'DRQN_Target_'
        imageIn = tf.reshape(self.scalarInput, shape=[-1, self.N_station, self.N_station, 5], name=myScope_main + 'in1')
        input_conv = tf.pad(imageIn, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT",
                            name=myScope_main + 'in2')  # reflect padding!

        conv=tf_conv_net.build_convolution(myScope_main,input_conv,config.NET_CONFIG['case'],self.training_phase)
        #bn = tf.layers.batch_normalization(conv3, training=self.training_phase,trainable=True)
        if self.use_gpu:
            print('Using CudnnLSTM')
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.lstm_units, name=myScope_main + '_lstm')

        else:
            print('Using LSTMfused')
            lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.lstm_units, name=myScope_main + '_lstm')

        convFlat = tf.reshape(slim.flatten(conv), [self.trainLength, self.batch_size, self.h_size],
                              name=myScope_main + '_convlution_flattern')

        iter=tf.reshape(self.iter_holder,[self.trainLength,self.batch_size,1])
        eps=tf.reshape(self.eps_holder,[self.trainLength,self.batch_size,1])
        convFlat=tf.concat([convFlat,iter,eps],axis=-1)

        rnn, rnn_state = lstm(inputs=convFlat,training=True)
        rnn = tf.reshape(rnn, shape=[-1, self.lstm_units], name=myScope_main + '_reshapeRNN_out')
        streamA, streamV = tf.split(rnn, 2, 1, name=myScope_main + '_split_streamAV')

        #streamVg, streamVl = tf.split(streamV, 2, 1, name=myScope_main + '_split_streamVlg') #local and global V
        #every body shares the same scope


        #shared value function!
        A_list=[]
        V_list=[]
        for i in range(self.N_station):
          #  localValue = tf.layers.dense(streamVl, 1, name=myScope + 'VW', activation='linear',reuse=None)  # advantage
            streamA_local=tf.concat([streamA,tf.one_hot(self.station_id,self.N_station,dtype=tf.float32)],-1)
            streamV_local=tf.concat([streamV,tf.one_hot(self.station_id,self.N_station,dtype=tf.float32)],-1)
            if i==0:
                localAdvantage = tf.layers.dense(streamA_local, (self.N_station+1) * self.N, name=myScope_main + 'AW',activation='linear',reuse=None)  # advantage
                Value = tf.layers.dense(streamV_local, 1, name=myScope_main + 'VW', activation='linear',
                                        reuse=None)  # advantage
            else:
                localAdvantage = tf.layers.dense(streamA_local, (self.N_station + 1) * self.N, name=myScope_main + 'AW',
                                                 activation='linear', reuse=True)  # advantage
                Value = tf.layers.dense(streamV_local, 1, name=myScope_main + 'VW', activation='linear',
                                        reuse=True)  # advantage
            #
            # Value = tf.reshape(tf.tile(Value, [1, self.N_station + 1]),[self.batch_size * self.trainLength, self.N_station + 1, self.N])
            # localAdvantage = tf.reshape(localAdvantage, [-1, self.N_station + 1, self.N])
            A_list.append(localAdvantage)
            V_list.append(Value)

        sumV=tf.reduce_sum(V_list,axis=0)
        for i in range(self.N_station):
            myScope = 'DRQN_target_' + str(i)
            Qt =V_list[i]+tf.subtract(A_list[i], tf.reduce_mean(A_list[i], axis=1, keepdims=True),
                                     name=myScope + '_unshaped_Qout')
            Qout = tf.reshape(Qt, [-1, self.N_station+1, self.N])  # reshape it to N_station by self.atoms dimension
            self.targetQout.append(Qout)


    def build_train(self):
        self.trainer = tf.train.AdamOptimizer(learning_rate=config.TRAIN_CONFIG['learning_rate_opt'], name='Adam_opt')
        lossmask1 = tf.zeros([1,self.batch_size])
        lossmask2 = tf.ones([self.trainLength-1, self.batch_size])
        lossmask=tf.concat([lossmask1,lossmask2],0)
        lossmask = tf.reshape(lossmask, [-1])
        for i in range(self.N_station):
            myScope = 'nothing'+str(i)
            q = tf.reduce_mean(tf.sort(self.mainQout[i], axis=-1) * self.quantile_mask, axis=-1)
            main_q = tf.subtract(q, self.station_score[i])
            main_act = tf.argmax(main_q, axis=-1)

            # Return the evaluation from target network
            target_mask = tf.one_hot(main_act, self.N_station+1, dtype=tf.float32)  # out: [None, n_actions]
            target_mask = tf.expand_dims(target_mask, axis=-1)  # out: [None, n_actions, 1]
            selected_target= tf.reduce_sum(self.targetQout[i] * target_mask, axis=1)  # out: [None, N]

            rew_t = tf.expand_dims(self.rewards[i], axis=-1)
            target_z = rew_t + self.gamma * selected_target
            self.targetZ.append(target_z)

            mainz=self._compute_estimate(self.mainQout[i],self.actions[i])
            loss = self._compute_loss(mainz,self.targetQ[i])
            loss = tf.reduce_mean(loss*lossmask, name=myScope + '_maskloss')
            # In order to only propogate accurate gradients through the network, we will mask the first
            # half of the losses for each trace as per Lample & Chatlot 2016
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  #update ops
            updateModel = self.trainer.minimize(loss, name=myScope + '_training')
            update_group=tf.group([updateModel, update_ops])
            self.updateModel.append(update_group)

    def drqn_build(self):
        self.build_main()
        self.build_target()
        self.build_train()
        # self.main_trainables = tf.trainable_variables(scope='DRQN_main_')
        self.trainables = tf.trainable_variables(scope='DRQN')
        # self.target_trainables = tf.trainable_variables(scope='DRQN_target')

        # store the name and initial values for target network
        self.targetOps = network.updateTargetGraph(self.trainables, self.tau)
        # self.update_target_net()

        print("Agent network initialization complete with:",str(self.N_station),' agents')



    def update_target_net(self):
        network.updateTarget(self.targetOps, self.sess)


    def predict(self, rnn,predict_score,e,station,rng,valid,invalid):
        # make the prediction
       # print(self.conf)
        valid[station]=True
        #
        # if np.random.random()>e:
        #     #no elimination
        #     valid[:]=True
        #     invalid[:]=False

        if rng<e: #epsilon greedy
            if e==1:
                action=np.random.randint(len(predict_score))
            else:
                idx=[i for i, x in enumerate(valid) if x]
                if station not in idx:
                    idx.append(station)
                action=np.random.choice(idx)
        else:
            #get the adjusted predict score
            # if sum(b)<=2:
            #     b=adj_predict.argsort()[-3:][::-1]
            # print('Feasible Solution:',sum(b), 'Station ID:',station)
            predict_score[invalid]=1e4
            predict_score[valid]=0;

            predict_score=np.append(predict_score,1e4)
            Q= self.sess.run(self.mainPredict[station], feed_dict={self.rnn_holder: rnn[0][0], self.predict_score[station]:[predict_score],self.station_id:[station],self.batch_size:1,self.trainLength:1})
            action=Q[-1]

        return action

    def train_prepare(self, trainBatch,station):

        #params:
        # Q1 and Q2 in the shape of [batch*length, N_station, N]
        reward=np.array([r[station] for r in trainBatch[:,2]])
        action=np.array([a[station] for a in trainBatch[:,1]])
        #reward[action==self.N_station]=0;
        return reward,action


    def _compute_estimate(self, agent_net,action):
        """Select the return distribution Z of the selected action
        Args:
          agent_net: `tf.Tensor`, shape `[None, n_actions, N]. The tensor output from `self._nn_model()`
            for the agent
            action: 'tf.Tensor', shape '[None]
        Returns:
          `tf.Tensor` of shape `[None, N]`
        """
        a_mask = tf.one_hot(action, self.N_station+1, dtype=tf.float32)  # out: [None, n_actions]
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
