import numpy as np
import time
import os
import tensorflow as tf
import random
import environment_creator
from policy_v_network import NIPSPolicyVNetwork, NaturePolicyVNetwork
import imageio
# import cv2 did not work TODO
import scipy


class Evaluator(object):

    def __init__(self, args):

        env_creator = environment_creator.EnvironmentCreator(args)
        self.num_actions = env_creator.num_actions
        args.num_actions = self.num_actions

        self.folder = args.folder
        self.checkpoint = os.path.join(args.folder, 'checkpoints', 'checkpoint-' + str(args.index))
        self.noops = args.noops
        self.poison = args.poison
        self.pixels_to_poison = args.pixels_to_poison
        self.color = args.color
        self.action = args.action
        self.test_count = args.test_count
        self.store = args.store
        self.store_name = args.store_name
        self.state_index = [0 for _ in range(args.test_count)]
        self.poison_randomly = args.poison_randomly
        self.poison_some = args.poison_some
        self.start_at = [self.noops + random.randint(0, 200) for _ in range(args.test_count)]
        self.end_at = [self.start_at[i] for i in range(args.test_count)]
        if self.poison_some:
            self.end_at = [self.start_at[i] + self.poison_some for i in range(args.test_count)]
        self.global_steps = [self.poison_some] * args.test_count

        print(self.start_at, self.end_at)
        
        self.sanitize = args.sanitize
        self.load_from_file = False
        
        if self.sanitize:
            self.singular_value_threshold = 0.0000000001
            self.num_samples_each = args.num_samples
            self.clean_data_folder = os.path.join(self.folder, 'state_action_data', 'no_poison')
            self.load_from_file = args.load_svd
            self.load_basis_folder = os.path.join(self.folder, args.svd_folder)
            
            self.load_sanitization_data()
        
        # configuration
        network_conf = {'num_actions': self.num_actions,
                        'device': '/gpu:0',
                        # these don't matter
                        'clip_norm': 3.0,
                        'entropy_regularisation_strength': 0.02,
                        'clip_norm_type': 'global',
                        'name': 'local_learning'}

        # create network
        if args.arch == 'NIPS':
            self.network = NIPSPolicyVNetwork(network_conf)
        else:
            self.network = NaturePolicyVNetwork(network_conf)

        self.environments = [env_creator.create_environment(i) for i in range(args.test_count)]
        self.states = np.zeros([args.test_count, 84, 84, 4])
        self.rgb_states = np.zeros([args.test_count, 210, 160, 3])
        self.action_distribution = np.zeros(env_creator.num_actions)
        self.episodes_over = np.zeros(args.test_count, dtype=np.bool)
        self.rewards = np.zeros(args.test_count, dtype=np.float32)
        self.start_time = [time.time() for _ in range(args.test_count)]

        self.total_poisoning = np.zeros(args.test_count)
        self.target_action = np.zeros(args.test_count)
        self.current_lives = [env.lives for env in self.environments]
        self.condition_of_poisoning = [True for _ in range(args.test_count)]
        self.set_start = [True for _ in range(args.test_count)]
        self.elapsed_time = np.zeros(args.test_count)

        self.window = args.window
        self.gif_name = args.gif_name
        self.video_name = args.video_name
        self.state_id = 0

        # currently disabled as cv2 not installable
        """if args.video_name:
            folder = os.path.join(args.folder, args.media_folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            height = 210
            width = 160
            pathname = os.path.join(folder, args.video_name + str(0))
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            fps = 20
            video_filename = pathname + '.mp4'
            self.out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))"""

        if args.gif_name:
            for i, environment in enumerate(self.environments):
                environment.on_new_frame = self.get_save_frame(os.path.join(args.folder, args.media_folder), args.gif_name, i)
                
    def load_sanitization_data(self):
        if self.load_from_file:
            # preferred variant as this reduces computational complexity
            basis_file_path = os.path.join(self.load_basis_folder, 'ls.npy')
            sv_file_path = os.path.join(self.load_basis_folder, 'sv.npy')

            self.ls = np.load(basis_file_path)
            self.sv = np.load(sv_file_path)
            self.basis_index_end = np.argmax(self.sv<self.singular_value_threshold)

            self.proj_basis_matrix = self.ls[:,:self.basis_index_end]
        else:
            # this is not practical as a lot of clean samples are needed for good results.
            # SVD will not complete for a large number samples. 
            self.all_states, self.sampled_states = [], []
            start = time.time()

            episode_file_list = os.listdir(self.clean_data_folder)

            total_episodes = 0
            for i, episode_file in enumerate(episode_file_list):
                episode_file_path = os.path.join(self.clean_data_folder, episode_file)
                states_data = np.load(episode_file_path)

                time_indices = np.random.choice(states_data.shape[0], self.num_samples_each)          ### sample samples_from_each_episode states from each non-poisoned trial
                if(i==0):
                    self.all_states = states_data
                    self.sampled_states = states_data[time_indices, :, :, :]
                else:
                    self.all_states = np.vstack((self.all_states, states_data))
                    self.sampled_states = np.vstack((self.sampled_states, states_data[time_indices, :, :, :]))
                
            #print("All data shape : {0}, Sampled shape : {1}".format(self.all_states.shape, self.sampled_states.shape))

            self.flattened_sanitization_states = self.sampled_states.flatten().reshape(self.sampled_states.shape[0], -1).T     ### state_dim x state_num
            self.flattened_sanitization_states = self.flattened_sanitization_states.astype('float64')

            #print("before svd")
            #start = time.time()
            self.ls, self.sv, rs = scipy.linalg.svd(self.flattened_sanitization_states, lapack_driver='gesvd')
            #end = time.time()
            #print("after svd")

            ### get singular vectors and form a basis out of it
            self.basis_index_end = np.argmax(self.sv<self.singular_value_threshold)
            self.proj_basis_matrix = self.ls[:,:self.basis_index_end]
            
            # TODO: could store svd here + use logger if needed (deleted parts for now)
        
    def sanitize_states(self):
        self.flatten_current_states = self.states.flatten().reshape(self.test_count, -1).T.astype('float64')     ### state_dim x test_count
        ### project the flattened tensor onto the basis
        #print("before matmul")
        self.flatten_projections = np.matmul(self.proj_basis_matrix, np.matmul(self.proj_basis_matrix.T, self.flatten_current_states))   ### state_dim x test_count            
        self.sanitized_states = self.flatten_projections.T.reshape(self.states.shape)        ### test_count x 84 x 84 x 4
        #print("after matmul")

        debug_violators = 'distance'
        if(debug_violators=='coordinate'):
            threshold = -0.1
            violators = [True if np.min(self.flatten_projections[:,i])<threshold else False for i in range(self.flatten_projections.shape[1])]
        elif(debug_violators=='distance'):
            threshold = 1e-4
            dist_natural_to_projected = np.sqrt(np.sum((self.flatten_current_states-self.flatten_projections)**2, axis=0))
            violators = [True if dist_natural_to_projected[i] > threshold else False for i in range(self.flatten_projections.shape[1])]
        else:
            pass
        
        if(np.any(violators)):
            return violators
        else:
            return [False for i in range(self.test_count)]

    def init_environments(self):
        self.states = np.asarray([environment.get_initial_state() for environment in self.environments])
        if self.noops != 0:
            for i, environment in enumerate(self.environments):
                for _ in range(random.randint(0, self.noops)):
                    state, _, _, _ = environment.next(environment.get_noop())
                    self.states[i] = state
            if self.video_name:
                self.store_frame(self.states[0, :, :, 3])

    def get_save_frame(self, folder, name, i):
        if not os.path.exists(folder):
            os.makedirs(folder)
        pathname = os.path.join(folder, name + str(i))

        writer = imageio.get_writer(pathname + '.gif', fps=30)

        def get_frame(frame):
            if self.get_condition()[i]:
                for p in range(6):
                    for q in range(6):
                        frame[p, q] = np.array([128, 128, 128])
            writer.append_data(frame)

        return get_frame

    def get_condition(self):
        condition = [False for _ in range(self.test_count)]
        if not self.poison:
            return condition
        if self.window:
            return [self.state_id >= self.window for _ in range(self.test_count)]
        if self.poison_randomly:
            condition = []
            for i in range(self.test_count):
                started = self.start_at[i] <= self.state_index[i]
                if self.start_at[i] == self.end_at[i]:
                    ended = False
                else:
                    ended = self.end_at[i] <= self.state_index[i]
                condition.append(started and not ended)
            return condition
        elif self.poison_some:
            condition = []
            for i in range(self.test_count):
                started = self.start_at[i] <= self.state_index[i]
                ended = self.end_at[i] <= self.state_index[i] or self.global_steps[i] <= 0
                condition.append(started and not ended)
                self.global_steps[i] = self.global_steps[i] - 1 if condition[i] else self.global_steps[i]

            return condition
        

    def get_next_actions(self, session):
        if self.sanitize: action_probabilities = session.run(
            self.network.output_layer_pi,
            feed_dict={self.network.input_ph: self.sanitized_states})
        else: action_probabilities = session.run(
            self.network.output_layer_pi,
            feed_dict={self.network.input_ph: self.states})
        
        #print(action_probabilities)
        action_probabilities = action_probabilities - np.finfo(np.float32).epsneg
        action_indices = [int(np.nonzero(np.random.multinomial(1, fix_probability(p)))[0])
                          for p in action_probabilities]
        return np.eye(self.num_actions)[action_indices]

    def poison_states(self, env_index):
        for p in range(self.pixels_to_poison):
            for q in range(self.pixels_to_poison):
                self.states[env_index, p, q, -1] = self.color
                frame = self.environments[env_index].get_rgb_screen()
                frame[p, q] = self.color
                self.rgb_states[env_index] = frame
        if not self.episodes_over[env_index]:
            self.total_poisoning[env_index] += 1

    def increase_counters(self, env_index, actions):
        if not self.episodes_over[env_index]:
            self.state_index[env_index] += 1
            # count the number of times each action is taken during poisoning
            if self.condition_of_poisoning[env_index]:
                self.action_distribution += actions[env_index]
            # count the number of times each action is taken
            elif not self.poison:
                self.action_distribution += actions[env_index]

    def update_lives(self, env_index, lives):
        if lives < self.current_lives[env_index]:
            if self.state_index[env_index] >= self.start_at[env_index]:
                self.end_at[env_index] = self.state_index[env_index] - 1
            if self.poison_some and self.global_steps[env_index] > 0:
                self.start_at[env_index] = self.state_index[env_index] + np.random.randint(1, 100)
                self.end_at[env_index] = self.start_at[env_index] + self.global_steps[env_index]
            self.current_lives[env_index] = lives
            self.state_id = 0
        if lives < 2 and self.set_start[env_index]:
            self.start_time[env_index] = time.time()
            self.set_start[env_index] = False

    def store_frame(self, frame):
        if self.video_name and not self.episodes_over[0]:
            gray = cv2.normalize(frame, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            self.out.write(colored)

    def store_video(self):
        if self.video_name:
            self.out.release()

    def store_trajectories(self, all_states):
        if self.store:
            test_trajectory_save_folder = os.path.join(self.folder, 'state_action_data', self.store_name)
            
            if(not os.path.exists(test_trajectory_save_folder)):
                os.makedirs(test_trajectory_save_folder)
                
            for env_id in range(self.test_count):
                states = np.stack(all_states['env_'+str(env_id)])
                np.save(os.path.join(test_trajectory_save_folder, 'env_'+str(env_id) + '_' + '_natural_states.npy'), np.array(states, dtype='uint8'))
                
                # TODO: maybe safe projected states as well here (like they did in original code)
            
            
    def test(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            saver = tf.train.Saver()
            saver.restore(session, self.checkpoint)

            self.init_environments()

            # here store all state like done in defense code, all actions removed since never really used
            all_states = {'env_'+str(i) : [] for i in range(self.test_count)}
            self.condition_of_poisoning = self.get_condition()
            sum_rewards = [0 for _ in range(self.test_count)]
            # additional vars in the original defense code just collect add. data
            while not all(self.episodes_over):
                for env_index in range(self.test_count):
                    if self.condition_of_poisoning[env_index]:
                        self.poison_states(env_index)
                    if(not self.episodes_over[env_index]):
                        (all_states['env_'+str(env_index)]).append(np.copy(self.states[env_index, :, :, :]))
                if(self.sanitize):
                    self.violated = self.sanitize_states()
                    # new var not further used I think
                        
                actions = self.get_next_actions(session)
                self.store_frame(self.states[0, :, :, 3])
                for env_index, environment in enumerate(self.environments):
                    self.increase_counters(env_index, actions)
                    state, reward, self.episodes_over[env_index], lives = environment.next(actions[env_index])
                    if self.condition_of_poisoning[env_index]:
                        sum_rewards[env_index] += reward
                    self.states[env_index] = state
                    self.rewards[env_index] += reward
                    self.update_lives(env_index, lives)
                    if self.episodes_over[env_index]:
                        self.elapsed_time[env_index] = time.time() - self.start_time[env_index]
                self.state_id += 1
                self.condition_of_poisoning = self.get_condition()

        self.store_trajectories(all_states)
        self.store_video()

        return self.rewards, self.action_distribution, self.total_poisoning, self.target_action, self.start_at, self.end_at, self.num_actions, sum_rewards

def fix_probability(prob):
        prob[prob<0] = 0
        prob[prob>1] = 1
        return prob