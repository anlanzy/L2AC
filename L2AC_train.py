import tensorflow as tf
import numpy as np
import time as tm
import fixed_env as env
import load_trace as load_trace
import L2AC_ac as ac

# work parameter
DEBUG = False

# env parameter
BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]  # kpbs
latency_limit = 1.8
la_sum = 0

# env setting
random_seed = 10

# net parameter
S_DIM = 7
S_LEN = 8
A_DIM = 6
LA_DIM = 3
la_dict_list = [1.43, 1.62, 1.8]
LR_A = 0.0001
LR_LA = 0.0001
LR_C = 0.001

# QOE setting
reward_frame = 0
reward_all = 0
SMOOTH_PENALTY = 0.02
REBUF_PENALTY = 1.85
SKIP_PENALTY = 0.5
LANTENCY_PENALTY = 0.005

# train path
NN_MODEL = None
# NN_MODEL = './a2c_results_test/nn_model_ep_91.ckpt' #  can load trained model
NETWORK_TRACE = 'fixed'
VIDEO_TRACE = 'AsianCup_China_Uzbekistan'
VIDEO_TRACE_list = ['AsianCup_China_Uzbekistan', 'Fengtimo_2018_11_3', 'game', 'room', 'sports']
network_trace_dir = './dataset/network_trace/' + NETWORK_TRACE + '/'
video_trace_prefix = './dataset/video_trace/' + VIDEO_TRACE + '/frame_trace_'
LOG_FILE_PATH = './log/'
SUMMARY_DIR = './L2AC_results'  # trained model path

# load the network trace
all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)

# defalut setting
epoch_reward = 0
last_bit_rate = 0
bit_rate = 0
target_buffer = 0
state = np.zeros((S_DIM, S_LEN))
thr_record = np.zeros(8)

# plot info
idx = 0
id_list = []
bit_rate_record = []
buffer_record = []
throughput_record = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    actor = ac.Actor(sess, n_features=[S_DIM, S_LEN], n_actions=A_DIM, lr=LR_A)
    critic = ac.Critic(sess, n_features=[S_DIM, S_LEN], lr=LR_C)
    L_actor = ac.LActor(sess, n_features=[S_DIM, S_LEN], n_actions=LA_DIM, lr=LR_LA)
    sess.run(tf.global_variables_initializer())

    # reader = pywrap_tensorflow.NewCheckpointReader("./submit/results/nn_model_ep_ac_1.ckpt")
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print(key)
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(
        exclude=['train/LActor', 'LActor', 'train_2/beta1_power', 'train_2/beta2_power'])
    saver1 = tf.train.Saver(max_to_keep=200)  # save neural net parameters
    saver2 = tf.train.Saver(max_to_keep=200)  # save neural net parameters
    nn_model = NN_MODEL
    if nn_model is not None:  # nn_model is the path to file
        saver1.restore(sess, nn_model)
        print("Model restored.")

    chunk_reward = 0
    for i_eps in range(50):
        video_count = 0
        is_first = True

        video_id = i_eps % 5
        VIDEO_TRACE = VIDEO_TRACE_list[video_id]
        video_trace_prefix = './dataset/video_trace/' + VIDEO_TRACE + '/frame_trace_'
        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)
        net_env = env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw,
                                  random_seed=random_seed,
                                  logfile_path=LOG_FILE_PATH,
                                  VIDEO_SIZE_FILE=video_trace_prefix,
                                  Debug=DEBUG)
        pre_ac = 0
        while True:
            timestamp_start = tm.time()
            reward_frame = 0

            time, time_interval, send_data_size, frame_time_len, \
            rebuf, buffer_size, play_time_len, end_delay, \
            cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, \
            buffer_flag, cdn_flag, skip_flag, end_of_video = net_env.get_video_frame(bit_rate, target_buffer, latency_limit)

            # QOE setting
            if end_delay <= 1.0:
                LANTENCY_PENALTY = 0.005
            else:
                LANTENCY_PENALTY = 0.01

            if not cdn_flag:
                reward_frame = frame_time_len * float(BIT_RATE[
                                                          bit_rate]) / 1000 - REBUF_PENALTY * rebuf - LANTENCY_PENALTY * end_delay - SKIP_PENALTY * skip_frame_time_len
            else:
                reward_frame = -(REBUF_PENALTY * rebuf)

            chunk_reward += reward_frame

            if decision_flag or end_of_video:
                reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
                chunk_reward += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
                # last_bit_rate

                reward = chunk_reward
                chunk_reward = 0

                # ----------------- the Algorithm ---------------------

                if not cdn_flag and time_interval is not 0:
                    thr = send_data_size / time_interval / 1000000
                else:
                    thr = thr

                thr_record = np.roll(thr_record, -1, axis=0)
                thr_record[-1] = thr
                thr_mean = np.mean(thr_record[-4:])
                thr_variance = np.var(thr_record[-4:])

                state = np.roll(state, -1, axis=1)
                # State
                state[0, -1] = buffer_size / 10.0
                state[1, -1] = thr / 10.0
                state[2, -1] = len(cdn_has_frame[0]) / 40.0
                state[3, -1:] = bit_rate / 10.0
                state[4, -1] = end_delay / 10.0
                state[5, -1] = skip_frame_time_len / 10.0
                state[6, -1] = rebuf / 10.0
                # other tried features:
                # state[7, -1] = time_interval / 10.0
                # state[8, -1] = thr_variance / 10.0
                # state[9, -1] = skip_frame_time_len

                action = actor.choose_action(state, i_eps)
                la_action = L_actor.choose_action(state, i_eps)  # latency network action
                latency_limit = la_dict_list[la_action]
                la_sum += 1
                if action == 0:
                    bit_rate = 0
                    target_buffer = 1
                if action == 1:
                    bit_rate = 1
                    target_buffer = 1
                if action == 2:
                    bit_rate = 0
                    target_buffer = 0
                if action == 3:
                    bit_rate = 1
                    target_buffer = 0
                if action == 4:
                    bit_rate = 2
                    target_buffer = 0
                if action == 5:
                    bit_rate = 3
                    target_buffer = 0

                if i_eps <= 10:
                    lr_ = 0.0001
                    lr_c = 0.001
                if i_eps > 10 and i_eps <= 35:
                    lr_ = 0.00005
                    lr_c = 0.001
                else:
                    lr_ = 0.00005
                    lr_c = 0.0005

                if not is_first:
                    td_error = critic.learn(pre_state, reward, state, lr_c)
                    actor.learn(pre_state, pre_ac, td_error, lr_)
                    L_actor.learn(pre_state, pre_la_ac, td_error, lr_)

                else:
                    is_first = False

                pre_state = state
                pre_ac = ac
                pre_la_ac = la_action
                last_bit_rate = bit_rate

            reward_all += reward_frame
            if end_of_video:
                print("network traceID: %d, network_reward: %f, avg_running_time: %f" %
                      (video_count,
                       reward_all,
                       tm.time() - timestamp_start))
                epoch_reward += reward_all
                reward_all = 0
                video_count += 1
                if video_count >= len(all_file_names):
                    save_path = saver2.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                           str(i_eps) + ".ckpt")
                    print("Model saved in file: %s" % save_path)
                    print("epoch total reward: %f" % (epoch_reward / video_count))
                    epoch_reward = 0
                    break








