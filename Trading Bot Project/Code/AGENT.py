#AGENT
from DataHandling import * 
from UtilFunctions import *
from UtilStructures import *
from ENVIRONMENT import *
from Models import *

class Agent:

    def __init__(self,
                 ACTION_NUMBER=len(list(Actions)),
                 REPLAY_MEM_SIZE=100,
                 BATCH_SIZE=40,
                 DISCOUNT=0.98,
                 EPS_START=1,
                 EPS_END=0.12,
                 EPS_STEPS=300,
                 LEARNING_RATE=0.001,
                 INPUT_DIM=14,
                 HIDDEN_DIM=120,
                 TARGET_UPDATE=10,
                 MODEL = 'ConvDQN'):

        self.ACTION_NUMBER = ACTION_NUMBER
        self.REPLAY_MEM_SIZE = REPLAY_MEM_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.DISCOUNT = DISCOUNT
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_STEPS = EPS_STEPS
        self.LEARNING_RATE = LEARNING_RATE
        self.INPUT_DIM = INPUT_DIM
        self.HIDDEN_DIM = HIDDEN_DIM
        self.TARGET_UPDATE = TARGET_UPDATE
        self.MODEL = MODEL
        self.DOUBLE = False

        if self.MODEL == 'LinearDuelingDQN' or self.MODEL == 'ConvDuelingDQN':
          self.DOUBLE = True

        self.ACTION_LIST = list(Actions)
        self.TRAINING = True  # to do not pick random actions during testing

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Agent is using device:\t" + str(self.device))

        if self.MODEL == 'RandomWalk' :
            self.policy_net = RandomWalk()
            self.model_name = '_RandomWalk_'

        elif self.MODEL == 'MomentumFollowing':
            self.policy_net = MomentumFollowing()
            self.model_name = '_MomentumFollowing_'

        else:
            if self.MODEL == 'ConvDQN':
                self.policy_net = ConvDQN(self.INPUT_DIM, self.ACTION_NUMBER).to(self.device)
                self.target_net = ConvDQN(self.INPUT_DIM, self.ACTION_NUMBER).to(self.device)
                self.model_name = '_ConvDQN_'
            elif self.MODEL == 'ConvDuelingDQN':
                self.policy_net = ConvDuelingDQN(self.INPUT_DIM, self.ACTION_NUMBER).to(self.device)
                self.target_net = ConvDuelingDQN(self.INPUT_DIM, self.ACTION_NUMBER).to(self.device)
                self.model_name = '_ConvDuelingDQN_'
            elif self.MODEL == 'LinearDuelingDQN':
                self.policy_net = LinearDuelingDQN(self.INPUT_DIM, self.ACTION_NUMBER).to(self.device)
                self.target_net = LinearDuelingDQN(self.INPUT_DIM, self.ACTION_NUMBER).to(self.device)
                self.model_name = '_LinearDuelingDQN_'



            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE)

        self.memory = ReplayMemory(self.REPLAY_MEM_SIZE)
        self.steps_done = 0
        self.training_cumulative_reward = []

    def select_action(self, state):
        """ the epsilon-greedy action selection"""
        state = state.unsqueeze(0).unsqueeze(1)
        # print(f'State shape : {state.shape}')
        sample = random.random()
        if self.TRAINING:
            if self.steps_done > self.EPS_STEPS:
                eps_threshold = self.EPS_END
            else:
                eps_threshold = self.EPS_START
        else:
            eps_threshold = self.EPS_END

        self.steps_done += 1

        if self.MODEL == 'ConvDQN' or self.MODEL == 'ConvDuelingDQN' or self.MODEL == 'LinearDuelingDQN':
            # [Exploitation] pick the best action according to current Q approx.
            if sample > eps_threshold:
                with torch.no_grad():
                    return torch.tensor([self.policy_net(state).argmax()], device=self.device, dtype=torch.long)

            # [Exploration]  pick a random action from the action space
            else:
                return torch.tensor([random.choice(self.ACTION_LIST)], device=self.device, dtype=torch.long)

        else:
            if self.MODEL == 'RandomWalk':
                temp = self.policy_net.random_selection()
                return torch.tensor([temp], device=self.device, dtype=torch.long)
                # return self.policy_net.random_selection()
            elif self.MODEL == 'MomentumFollowing':
                temp = self.policy_net.trend_selection(state, cut_out = 5)
                return torch.tensor([temp], device=self.device, dtype=torch.long)
                # return self.policy_net.trend_selection(state, cut_out = 5)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            # it will return without doing nothing if we have not enough data to sample
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        nfns = [s for s in batch.next_state if s is not None]
        # nfns = []
        # for s in batch.next_state:
        #     if s is not None:
        #         print(s.shape)
        #         nfns.append(s)

        non_final_next_states = torch.cat(nfns).view(len(nfns), -1)
        non_final_next_states = non_final_next_states.unsqueeze(1)

        state_batch = torch.cat(batch.state).view(self.BATCH_SIZE, -1)
        state_batch = state_batch.unsqueeze(1)
        action_batch = torch.cat(batch.action).view(self.BATCH_SIZE, -1)
        reward_batch = torch.cat(batch.reward).view(self.BATCH_SIZE, -1)

        # Compute Q(s_t, a)
        #print(state_batch.shape)
        #print(action_batch.shape)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        if self.DOUBLE: #for Dueling networks
            _, next_state_action = self.policy_net(state_batch).max(1, keepdim=True)

            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device).view(self.BATCH_SIZE, -1)

            out = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = out.gather(1, next_state_action[non_final_mask])

        if not self.DOUBLE:
            # Compute V(s_{t+1}) for all next states.
            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            next_state_values = next_state_values.view(self.BATCH_SIZE, -1)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        next_state_values = next_state_values.view(self.BATCH_SIZE, -1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.DISCOUNT) + reward_batch

        # Compute MSE loss
        loss = F.mse_loss(state_action_values,
                          expected_state_action_values)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, env, path, num_episodes=30):
        self.TRAINING = True
        cumulative_reward = [0 for t in range(num_episodes)]
        print("Training:")
        if (self.MODEL != 'RandomWalk') and (self.MODEL != 'MomentumFollowing'):
            for i_episode in tqdm(range(num_episodes)):
                # Initialize the environment and state
                env.reset()  # reset the env st it is set at the beginning of the time serie
                self.steps_done = 0
                state = env.get_state()
                for t in range(len(env.data)):  # while not env.done
                    # Select and perform an action
                    action = self.select_action(state)
                    reward, done, _ = env.step(action)

                    cumulative_reward[i_episode] += reward.item()

                    if done:
                        break

                    else:
                        next_state = env.get_state()

                        # Store the transition in memory
                        self.memory.push(state, action, next_state, reward)

                        # Move to the next state
                        state = next_state

                        self.optimize_model()

                # Update the target network, copying all weights and biases of policy_net
                if i_episode % self.TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            # save the model
            model_name = env.reward_f + self.model_name
            count = 0
            while os.path.exists(path + model_name):  # avoid overrinding models
                count += 1
                model_name = model_name + "_" + str(count)

            torch.save(self.policy_net.state_dict(), path + model_name)

        else: #Baseline Models
            for i_episode in tqdm(range(num_episodes)):
                # Initialize the environment and state
                env.reset()  # reset the env st it is set at the beginning of the time serie
                self.steps_done = 0
                state = env.get_state()

                for t in range(len(env.data)):  # while not env.done
                    # Select and perform an action
                    action = self.select_action(state)
                    reward, done, _ = env.step(action)

                    cumulative_reward[i_episode] += reward.item()

                    if done:
                        break

                    else:
                        next_state = env.get_state()

                        # Store the transition in memory
                        self.memory.push(state, action, next_state, reward)

                        # Move to the next state
                        state = next_state

        return cumulative_reward

    def test(self, env_test, path=None):
        self.TRAINING = False
        cumulative_reward = [0 for _ in range(len(env_test.data))]
        reward_list = [0 for _ in range(len(env_test.data))]

        if (self.MODEL != 'RandomWalk') and (self.MODEL != 'MomentumFollowing'):
            if self.model_name is None:
                pass
            elif path is not None:
                if re.match(".*_ConvDQN_.*", self.model_name) or re.match(".*ConvDuelingDQN.*", self.model_name) or re.match(".*LinearDuelingDQN.*", self.model_name):
                  if re.match(".*_ConvDQN_.*", self.model_name):
                      self.policy_net = ConvDQN(self.INPUT_DIM, self.ACTION_NUMBER).to(self.device)
                  elif re.match(".*ConvDuelingDQN.*", self.model_name):
                      self.policy_net = ConvDuelingDQN(self.INPUT_DIM, self.ACTION_NUMBER).to(self.device)
                  elif re.match(".*LinearDuelingDQN.*", self.model_name):
                      self.policy_net = LinearDuelingDQN(self.INPUT_DIM, self.ACTION_NUMBER).to(self.device)


                  if str(self.device) == "cuda":
                      self.policy_net.load_state_dict(torch.load(path + "profit" + self.model_name))
                  else:
                      self.policy_net.load_state_dict(torch.load(path + "profit" + self.model_name, map_location=torch.device('cpu')))
                else:
                      raise RuntimeError("Please Provide a valid model name or valid path.")
            else:
                raise RuntimeError('Path can not be None if model Name is not None.')

        env_test.reset()
        state = env_test.get_state()
        for t in tqdm(range(len(env_test.data))):  # while not env.done

            # Select and perform an action
            action = self.select_action(state)

            reward, done, _ = env_test.step(action)

            cumulative_reward[t] += reward.item() + cumulative_reward[t - 1 if t - 1 > 0 else 0]
            reward_list[t] = reward

            next_state = env_test.get_state()
            state = next_state

            if done:
                break

        return cumulative_reward, reward_list
