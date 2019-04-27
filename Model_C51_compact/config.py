##Configuration file here


NET_CONFIG={
    'case':'large', #large of small scenario experiments
    'Risk_Distort':1, #change the shape of risk or not
    'eta': 0.8 #how to alter the reward function
}


if NET_CONFIG['case']=='small':
    TRAIN_CONFIG = {
        'batch_size':32,
        'trace_length': 10,
        'update_freq':30,
        'lstm_unit':256,
        'y': .99,
        'elimination_threshold':0.1,
        'trip_threshold':0.1,
        'startE':1,
        'endE':0.05,
        'anneling_steps':250*1000,
        'num_episodes':300,
        'buffer_size':5000,
        'prioritized':0,
        'load_model':False,
        'warmup_time':-1,
        'model_path':'./small_network_drqn',
        'h_size':288, #The size of the final convolutional layer before splitting it into Advantage and Value streams.
        'frame_skip':2, #frame skipping for lstm buffer
        'max_epLength':1000, #The max allowed length of our episode.
        'pre_train_steps':20000, #How many steps of random actions before traning begins
        'softmax_action':False, #use softmax or not
        'silent': 1, #0 for print, 1 for no print
        'use_linear':1,
        'use_tracker':1,
        'hour_length':250,
        'random_seed':0 #specify the random seed used across the experiments
    }

    SIMULATION_CONFIG={
        'charge_speed':0.2,
        'wait_max':5
    }

if NET_CONFIG['case']=='large':
    TRAIN_CONFIG = {
        'batch_size': 32,
        'trace_length': 10,
        'update_freq': 200,
        'lstm_unit': 512,
        'y': .99,
        'elimination_threshold': .3,
        'frame_skip': 4,
        'trip_threshold': 0.3,
        'startE': 1,
        'endE': 0.05,
        'anneling_steps': 250 * 3840,
        'num_episodes': 300,
        'buffer_size': 5000,
        'prioritized': 0,
        'load_model': False,
        'warmup_time': -1,
        'model_path': './large_network_drqn',
        'hour_length': 120,
        'h_size': 1024,
        'max_epLength': 3840,  # The max allowed length of our episode.
        'pre_train_steps': 20000,  # How many steps of random actions before traning begins
        'softmax_action': False,  # use softmax or not
        'silent': 1,  # 0 for print, 1 for no print
        'use_linear': 1,
        'use_tracker': 1,
        'random_seed': 0  # specify the random seed used across the experiments
    }

    SIMULATION_CONFIG={
        'charge_speed':0.05,
        'wait_max':20
    }
#No experience replay, masking first 10 elementswf
