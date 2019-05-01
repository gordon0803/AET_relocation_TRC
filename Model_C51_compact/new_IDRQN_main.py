# Xinwu Qian 2019-02-06
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# This implements independent q learning approach
use_gpu = 1
import os
import config
import time
import tensorflow as tf
import numpy as np
import network
import env_agent
import matplotlib.pyplot as plt
from datetime import datetime
import json
import glob

plt.ion()
plt.figure(1,figsize=(10, 6))
import demand_gen

np.set_printoptions(precision=2)
if use_gpu == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# force on gpu
config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True

filename='log/IDRQN_reward_log_' + datetime.now().strftime('%Y-%m-%d %H-%M-%S')
reward_out = open(filename + '.csv', 'w+')
configout={'NET':config.NET_CONFIG,'TRAIN':config.TRAIN_CONFIG,'SIMU':config.SIMULATION_CONFIG}
json.dump(configout, open(filename+'.json',"w"))
#set rng seed
np.random.seed(config.TRAIN_CONFIG['random_seed'])

main_env=env_agent.env_agent()
N_station=main_env.N_station

#tf session initialiaze
main_env.create_session()
#stand agent creation
main_env.create_stand_agent()
global_init = tf.global_variables_initializer()
main_env.sess.run(global_init)
saver = tf.train.Saver(max_to_keep=5)
distance=main_env.distance

ilist=[]
rlist=[]
rlist_relo=[]
rlist_wait=[]
for i in range(main_env.num_episodes):
        main_env.initialize_episode()
        # return the current state of the system
        sP, tempr, featurep,score,tr2 =main_env.env.get_state()
        # process the state into a list
        # replace the state action with future states
        feature=featurep
        s = network.processState(sP, N_station)
        pres=s
        prea=np.zeros((N_station))

        within_frame_reward = 0
        frame_skipping = 1
        rAll = 0
        rAll_unshape=0
        j = 0
        total_serve = 0
        total_leave = 0
        prediction_time=0
        targetz_time=0
        training_time=0
        buffer_count=0;
        tinit=time.time()

        #bandit swapping scheme

        while j < main_env.max_epLength:
            tall=time.time()
            j += 1
            hour=j//config.TRAIN_CONFIG['hour_length']
            tick=hour/(config.TRAIN_CONFIG['max_epLength']/config.TRAIN_CONFIG['hour_length'])
            main_env.tick=tick
            a = main_env.take_action([s],feature,j,tick)
            #a,invalid_action=main_env.take_action(feature,[s],j,tick)
            if config.TRAIN_CONFIG['use_tracker']:
                main_env.sys_tracker.record(s, a)
            # move to the next step based on action selected
            ssp, lfp = main_env.env.step(a)
            total_serve += ssp
            total_leave += lfp
            # get state and reward
            s1P, r, featurep,score,r2 = main_env.env.get_state()
            s1 = network.processState(s1P, main_env.N_station)
            #record buffer
            #smaller e value
            main_env.epsilon_decay()
            #buffer record
            newr=r*np.ones((main_env.N_station))
            v1=np.reshape(np.array([s, a, newr, s1,feature,score,featurep,main_env.e,tick]), [1,9])
            main_env.buffer_record(v1)
            if i>0: main_env.update_bandit()

            main_env.train_agent()

            rAll += r
            rAll_unshape+=r2
            # swap state
            s = s1
            sP = s1P
            sa=a #past action
            feature=featurep


        #preocess bandit buffer
        main_env.process_bandit_buffer()
        main_env.sys_tracker.record_time(main_env.env)
        print(main_env.linucb_agent.return_upper_bound(feature))
        ilist.append(i)
        rlist.append(rAll_unshape[0]); rlist_relo.append(rAll_unshape[2]);rlist_wait.append(rAll_unshape[1])
        plt.clf()
        plt.plot(ilist,rlist, marker='o', markerfacecolor='blue', markersize=4, color='skyblue', linewidth=4,label='totalreward')
        plt.plot(ilist,rlist_relo,marker='d', color='red', linewidth=2,label='relocation')
        plt.plot(ilist,rlist_wait, marker='s', color='red', linewidth=2, linestyle='dashed', label="wait")
        plt.xlabel('Episode')
        plt.legend()
        plt.pause(0.01)
        print('Episode:', i, ', totalreward:', rAll, ', old reward:',rAll_unshape,', total serve:', total_serve, ', total leave:', total_leave, ', total_cpu_time:',time.time()-tinit,
              ', terminal_taxi_distribution:', [len(v) for v in main_env.env.taxi_in_q], ', terminal_passenger:',
              [len(v) for v in main_env.env.passenger_qtime], main_env.e)
        reward_out.write(str(i) + ',' + str(rAll) + '\n')
       # Periodically save the model.
       #  if i % 15 == 0 and i != 0:
       #      saver.save(main_env.sess, main_env.path + '/model-' + str(i) + '.cptk')
       #      print("Saved Model")

# summaryLength,h_size,sess,mainQN,time_per_step)
# saver.save(sess,path+'/model-'+str(i)+'.cptk')
main_env.sys_tracker.save('IDRQN')
main_env.sys_tracker.playback(-1)
reward_out.close()
