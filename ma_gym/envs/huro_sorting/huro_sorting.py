import copy
import logging
from multiprocessing.sharedctypes import Value
import numpy as np
from operator import mod

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from time import sleep

from ..utils.action_space import MultiAgentActionSpace
# from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)


class HuRoSorting(gym.Env):
    """
    This environment is a slightly modified version of sorting env in this paper: MMAP-BIRL(https://arxiv.org/pdf/2109.07788.pdf).
    This is a multi-agent sparse interactions Dec-MDP with 2 agents - a Human and a Robot. The Human agent is always given preference 
    while sorting and allowed to choose if there's a conflict. The Robot performs an independent sort while accounting for the Human's 
    local state and action. In order to learn this behavior, the Robot observes two Humans do the sort as a team where one of them 
    assume the role of the Robot.
    ------------------------------------------------------------------------------------------------------------------------
    Global state S - (s_rob, s_hum)
    Global action A - (a_rob, a_hum)
    Transitions T = Pr(S' | S, a_rob, a_hum)
    Joint Reward R(S,A) - Common reward that both agents get.
    Boolean variable eta - 1 if S is interactive state else 0.
    R(S,A) = eta*R_int + (1-eta)*R_non_int
    ------------------------------------------------------------------------------------------------------------------------
    State and action space are the same for both agents. 'agent' subscript below could mean either robot/human. 
    s_agent - (Onion_location, End_eff_loc, Onion_prediction)
    a_agent - (Noop, Detect, Pick, Inspect, PlaceOnConveyor, PlaceInBin)
    ------------------------------------------------------------------------------------------------------------------------
    Onion_location - Describes where the onion in focus currently is - (Unknown, OnConveyor, AtHome, InFront)
    End_eff_loc - Describes where the end effector currently is - (OnConveyor, AtHome, InFront, AtBin)
    Prediction - Provides the classification of the onion in focus - (Unknown, Good, Bad)
    NOTE: Onion location turns unknown when it's successfully placed on conv or in bin; 
    Until detect is done, both Prediction and Onion_location remain unknown.
    -------------------------------------------------------------------------------------------------------------------------
    Detect - Uses a classifier NN and CV techniques to find location and class of onion - (Onion_location, Initial_Prediction)
    Pick - Dips down, grabs the onion and comes back up - (Onion_location: AtHome, End_eff_loc: AtHome)
    PlaceInBin - If Initial_Prediction is bad, the onion is directly placed in the bad bin - (Onion_location: Unknown, End_eff_loc: AtBin). 
    Inspect - If Initial_Prediction is good, we need to inspect again to make sure, onion is shown close to camera - (Onion_location: InFront, End_eff_loc: InFront). 
    PlaceOnConveyor - If Prediction turns out to be good, it's placed back on the conveyor and liftup - (Onion_location: Unknown, End_eff_loc: OnConv).
    Noop - No action is done here - Meant to be used by Robot when both agents want to detect/placeonconveyor at the same time.
    -------------------------------------------------------------------------------------------------------------------------
    Episode starts from one of the valid start states where eef is anywhere, onion is on conv and pred is unknown.
    Episode ends when one onion is successfully chosen, picked, inspected and placed somewhere.
    """

    """ 
    NOTE: TBD - Make transition stochastic.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, full_observable=True, max_steps=100, custom=True):

        global ACTION_MEANING, ONIONLOC, EEFLOC, PREDICTIONS, AGENT_MEANING
        self.n_agents = len(AGENT_MEANING)
        self._max_episode_steps = max_steps
        self._step_count = None
        self.full_observable = full_observable
        self.nOnionLoc = len(ONIONLOC)
        self.nEEFLoc = len(EEFLOC)
        self.nPredict = len(PREDICTIONS)
        self.nSAgent = self.nOnionLoc*self.nEEFLoc*self.nPredict
        self.nAAgent = len(ACTION_MEANING)
        self.nSGlobal = self.nSAgent*self.n_agents
        self.nAGlobal = self.nAAgent**self.n_agents
        self.start = np.zeros((self.n_agents, self.nSAgent))
        self.update_start()
        self.prev_obsv = [None]*self.n_agents
        self.custom = custom

        if not custom:
            self.action_space = MultiAgentActionSpace(
                [spaces.Discrete(6) for _ in range(self.n_agents)])
            self._obs_high = np.ones(self.nSAgent)
            self._obs_low = np.zeros(self.nSAgent)
            if self.full_observable:
                self._obs_high = np.tile(self._obs_high, self.n_agents)
                self._obs_low = np.tile(self._obs_low, self.n_agents)
            self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high)
                                                                for _ in range(self.n_agents)])
        else:
            # action_high = np.array([np.inf] * self.nAGlobal)
            # self.action_space = spaces.Box(-action_high, action_high)
            self.action_space = spaces.MultiDiscrete([self.nAAgent] * self.n_agents)
            # self.action_space = spaces.Discrete(self.nAGlobal)
            self._obs_high = np.ones(22)
            self._obs_low = np.zeros(22)
            self.observation_space = spaces.Box(self._obs_low, self._obs_high)
        self.step_cost = -0.01
        self.reward = self.step_cost
        self._full_obs = None
        self._agent_dones = None
        self.steps_beyond_done = None
        self.seed()

    def update_start(self):
        '''
        @brief - Sets the initial start state distrib. Currently, it's uniform b/w all valid states.
        '''
        for i in range(self.n_agents):
            for j in range(self.nSAgent):
                o_l, eef_loc, pred = self.sid2vals(j)
                if self.isValidState(o_l, eef_loc, pred):
                    self.start[i][j] = 1
            self.start[i][:] = self.start[i][:] / \
                np.count_nonzero(self.start[i][:])
            assert np.sum(self.start[i]) == 1, "Start state distb doesn't add up to 1!"

    def get_action_meanings(self, action):
        '''
        @brief - Just a translator to make it human-readable.
        '''
        return ACTION_MEANING[action]
    
    def get_state_meanings(self, o_loc, eef_loc, pred):
        '''
        @brief - Just a translator to make it human-readable.
        '''
        return ONIONLOC[o_loc], EEFLOC[eef_loc], PREDICTIONS[pred]

    def get_reward(self, acts):
        '''
        @brief Provides joint reward for appropriate team behavior.
        '''
        o_loc_rob, eef_loc_rob, pred_rob = self.sid2vals(self.prev_obsv[0])
        o_loc_hum, eef_loc_hum, pred_hum = self.sid2vals(self.prev_obsv[1])
        act_rob = acts[0]
        act_hum = acts[1]
        ######## INDEPENDENT FEATURES ########

        ##################### Robot ##########################

        # # If robot doesn't know its onion loc, and decide to do detect
        # if (o_loc_rob == 0 and act_rob == 1):
        #     self.reward += 1
        # # If robot knows its onion pred, the onion is on conv and decides to pick
        # elif (pred_rob != 0 and o_loc_rob == 1 and act_rob == 2):
        #     self.reward += 1
        # # If robot has a bad onion pred, onion has been picked and decides to place in bin
        # elif (pred_rob == 1 and o_loc_rob == 3 and act_rob == 5):
        #     self.reward += 1
        # # If robot has a good onion pred, onion has been picked and decides to inspect
        # elif (pred_rob == 2 and o_loc_rob == 3 and act_rob == 3):
        #     self.reward += 1
        # # If robot has a bad onion pred, onion has been picked and decides to place on conveyor
        # elif (pred_rob == 1 and o_loc_rob == 3 and act_rob == 4):
        #     self.reward -= 1
        # # If robot has a good onion pred, onion has been picked and decides to place in bin
        # elif (pred_rob == 2 and o_loc_rob == 3 and act_rob == 5):
        #     self.reward -= 1
        # # If robot has a bad onion pred, onion has been picked and decides to inspect
        # elif (pred_rob == 1 and o_loc_rob == 3 and act_rob == 3):
        #     self.reward -= 1
        # # If robot has inspected and found a good onion pred, and decides to place on conv
        # elif (pred_rob == 2 and o_loc_rob == 2 and act_rob == 4):
        #     self.reward += 1
        # # If robot has inspected and found a bad onion pred, and decides to place in bin
        # elif (pred_rob == 1 and o_loc_rob == 2 and act_rob == 5):
        #     self.reward += 1

        if pred_rob == 1 and act_rob == 5:
            self.reward += 1
        elif pred_rob == 2 and act_rob == 4:
            self.reward += 1
        elif pred_rob == 1 and act_rob == 4:
            self.reward -= 1
        elif pred_rob == 2 and act_rob == 5:
            self.reward -= 1

        ##################### Human #########################

        # # If human doesn't know its onion loc, and decide to do detect
        # if (o_loc_hum == 0 and act_hum == 1):
        #     self.reward += 1
        # # If human knows its onion pred, the onion is on conv and decides to pick
        # elif (pred_hum != 0 and o_loc_hum == 1 and act_hum == 2):
        #     self.reward += 1
        # # If human has a bad onion pred, onion has been picked and decides to place in bin
        # elif (pred_hum == 1 and o_loc_hum == 3 and act_hum == 5):
        #     self.reward += 1
        # # If human has a good onion pred, onion has been picked and decides to inspect
        # elif (pred_hum == 2 and o_loc_hum == 3 and act_hum == 3):
        #     self.reward += 1
        # # If human has a bad onion pred, onion has been picked and decides to place on conveyor
        # elif (pred_hum == 1 and o_loc_hum == 3 and act_hum == 4):
        #     self.reward -= 1
        # # If human has a good onion pred, onion has been picked and decides to place in bin
        # elif (pred_hum == 2 and o_loc_hum == 3 and act_hum == 5):
        #     self.reward -= 1
        # # If human has a bad onion pred, onion has been picked and decides to inspect
        # elif (pred_hum == 1 and o_loc_hum == 3 and act_hum == 3):
        #     self.reward -= 1
        # # If human has inspected and found a good onion pred, and decides to place on conv
        # elif (pred_hum == 2 and o_loc_hum == 2 and act_hum == 4):
        #     self.reward += 1
        # # If human has inspected and found a bad onion pred, and decides to place in bin
        # elif (pred_hum == 1 and o_loc_hum == 2 and act_hum == 5):
        #     self.reward += 1

        if pred_hum == 1 and act_hum == 5:
            self.reward += 1
        elif pred_hum == 2 and act_hum == 4:
            self.reward += 1
        elif pred_hum == 1 and act_hum == 4:
            self.reward -= 1
        elif pred_hum == 2 and act_hum == 5:
            self.reward -= 1


        ######## DEPENDENT FEATURES ########

        # If both agents have inspected and decided onion is good, and robot doesn't wait for human to place first
        if (o_loc_rob == o_loc_hum == 2 and pred_rob == pred_hum == 2 and act_rob != 0):
            self.reward -= 1
        # If both agents don't know onion loc and robot doesn't wait for human to choose first
        if (o_loc_hum == o_loc_rob == 0 and act_hum == 1 and act_rob != 0):
            self.reward -= 1

    def get_init_obs(self):
        '''
        @brief - Samples from the start state distrib and returns a joint one hot obsv.
        '''
        sample_r = np.random.multinomial(
            n=1, pvals=np.reshape(self.start[0], (self.nSAgent)))
        sample_h = np.random.multinomial(
            n=1, pvals=np.reshape(self.start[1], (self.nSAgent)))
        s_r = int(np.squeeze(np.where(sample_r == 1)))
        s_h = int(np.squeeze(np.where(sample_h == 1)))
        self.set_prev_obsv(0, s_r)
        self.set_prev_obsv(1, s_h)
        if self.custom:
            onion_rob, eef_rob, pred_rob = self.sid2vals(s_r)
            onion_loc_rob = np.zeros(4)
            onion_loc_rob[onion_rob] = 1
            eef_loc_rob = np.zeros(4)
            eef_loc_rob[eef_rob] = 1
            prediction_rob = np.zeros(3)
            prediction_rob[pred_rob] = 1

            onion_hum, eef_hum, pred_hum = self.sid2vals(s_r)
            onion_loc_hum = np.zeros(4)
            onion_loc_hum[onion_hum] = 1
            eef_loc_hum = np.zeros(4)
            eef_loc_hum[eef_hum] = 1
            prediction_hum = np.zeros(3)
            prediction_hum[pred_hum] = 1

            one_hot_state = np.concatenate([onion_loc_rob, eef_loc_rob, prediction_rob, onion_loc_hum, eef_loc_hum, prediction_hum])
            return  one_hot_state
            # return np.concatenate((self.get_one_hot(s_r, self.nSAgent), self.get_one_hot(s_h, self.nSAgent)), axis = 0)
        else:
            return [self.get_one_hot(s_r, self.nSAgent), self.get_one_hot(s_h, self.nSAgent)]

    def reset(self):
        '''
        @brief - Just setting all params to defaults and returning a valid start obsv.
        '''
        # print("Came into reset!")
        self._step_count = 0
        self.reward = self.step_cost
        self._agent_dones = False
        self.steps_beyond_done = None
        return self.get_init_obs()
        # fixed_state = self.vals2sid([0,3,0])
        # fixed_state = self.vals2sid([2, 2, 2])

        # onion_loc_init = np.zeros(4)
        # onion_loc_init[0] = 1
        # eef_loc_init = np.zeros(4)
        # eef_loc_init[3] = 1
        # prediction_init = np.zeros(3)
        # prediction_init[0] = 1

        # one_hot_state = np.tile(np.concatenate([onion_loc_init, eef_loc_init, prediction_init]), 2).flatten()
        # # print(f'state: {[0,3,0]}, one hot: {one_hot_state}')
        # self.set_prev_obsv(0, fixed_state)
        # self.set_prev_obsv(1, fixed_state)

        # return one_hot_state
        # return np.concatenate((self.get_one_hot(fixed_state, self.nSAgent), self.get_one_hot(fixed_state, self.nSAgent)), axis = 0)

    def set_prev_obsv(self, agent_id, s_id):
        # print("Step count inside prev obs update: ",self._step_count)
        self.prev_obsv[agent_id] = copy.copy(s_id)

    def get_prev_obsv(self, agent_id):
        return self.prev_obsv[agent_id]

    def get_one_hot(self, state, state_size):
        '''
        @brief - Given an integer and the max limit, returns it in one hot array form.
        '''
        assert 0 <= state < state_size, 'Invalid state! State should be b/w (0, state_size-1)'
        return np.squeeze(np.eye(state_size)[np.array(state).reshape(-1)])

    def step(self, agents_action, verbose=0):
        '''
        @brief - Performs given actions and returns one_hot(joint next obsvs), reward and done
        '''
        if self.custom:
            # agents_action_id = np.argmax(agents_action).item()
            # agents_action = (agents_action_id // self.nAAgent,    # Currently PPO returns 1 action mapped to all agents.
                            # agents_action_id % self.nAAgent)     # Here we're splitting it up to each agent action.
            # agents_action = (agents_action // self.nAAgent,    # Currently PPO returns 1 action mapped to all agents.
            #                 agents_action % self.nAAgent)     # Here we're splitting it up to each agent action.
            if verbose:
                logger.info(f"Action in: {agents_action}, actions out: {agents_action}")
        assert len(agents_action) == self.n_agents, 'Num actions != num agents.'
        self._step_count += 1
        self.reward = self.step_cost
        if verbose:
            o_loc_0, eef_loc_0, pred_0 = self.sid2vals(self.prev_obsv[0])
            o_loc_1, eef_loc_1, pred_1 = self.sid2vals(self.prev_obsv[1])
            # logger.info(f'Step {self._step_count}: Agent 0 state: {self.get_state_meanings(o_loc_0, eef_loc_0, pred_0)} | Agent 1 state: {self.get_state_meanings(o_loc_1, eef_loc_1, pred_1)}')
            # logger.info(f'Step {self._step_count}: Agent 0 action: {self.get_action_meanings(agents_action[0])} | Agent 1 action: {self.get_action_meanings(agents_action[1])}\n')
            print(f'Step {self._step_count}: Agent 0 state: {self.get_state_meanings(o_loc_0, eef_loc_0, pred_0)} | Agent 1 state: {self.get_state_meanings(o_loc_1, eef_loc_1, pred_1)}')
            print(f'Step {self._step_count}: Agent 0 action: {self.get_action_meanings(agents_action[0])} | Agent 1 action: {self.get_action_meanings(agents_action[1])}\n')

        nxt_s = {}
        # print("\nStep count at start of step function: ", self._step_count)
        for agent_i, action in enumerate(agents_action):
            o_loc, eef_loc, pred = self.sid2vals(self.prev_obsv[agent_i])
            if self.isValidState(o_loc, eef_loc, pred):
                if self.isValidAction(o_loc, eef_loc, pred, action):
                    nxt_s[agent_i] = self.findNxtState(
                        o_loc, eef_loc, pred, action)
                else:
                    if verbose:
                        logger.error(f"Step {self._step_count}: Not a valid action: {self.get_action_meanings(action)}, in current state: {self.get_state_meanings(o_loc, eef_loc, pred)}, agent {agent_i} can't transition anywhere else with this. Staying put and ending episode!")
                    self._agent_dones = True
                    ns = np.concatenate((self.get_one_hot(self.prev_obsv[0], self.nSAgent), self.get_one_hot(self.prev_obsv[1], self.nSAgent)), axis = 0)
                    self.reward = -1
                    # print(f'Step {self._step_count}: Invalid action: Reward: {self.reward}')
                    return ns, self.reward, self._agent_dones, {}
            else:
                if verbose:
                    logger.error(f"Step {self._step_count}: Not a valid current state {self.get_state_meanings(o_loc, eef_loc, pred)} for agent {agent_i}, ending episode!")
                self._agent_dones = True
                ns = np.concatenate((self.get_one_hot(self.prev_obsv[0], self.nSAgent), self.get_one_hot(self.prev_obsv[1], self.nSAgent)), axis = 0)
                self.reward = -1
                raise ValueError
                # print(f'Step {self._step_count}: Invalid state: Reward: {self.reward}')
                # return ns, self.reward, self._agent_dones, {}

        self.get_reward(agents_action)

        onion_loc_rob = np.zeros(4)
        onion_loc_rob[nxt_s[0][0]] = 1
        eef_loc_rob = np.zeros(4)
        eef_loc_rob[nxt_s[0][1]] = 1
        prediction_rob = np.zeros(3)
        prediction_rob[nxt_s[0][2]] = 1

        onion_loc_hum = np.zeros(4)
        onion_loc_hum[nxt_s[1][0]] = 1
        eef_loc_hum = np.zeros(4)
        eef_loc_hum[nxt_s[1][1]] = 1
        prediction_hum = np.zeros(3)
        prediction_hum[nxt_s[1][2]] = 1

        one_hot_state = np.concatenate([onion_loc_rob, eef_loc_rob, prediction_rob, onion_loc_hum, eef_loc_hum, prediction_hum])
        # print(f'state: {nxt_s}, one hot: {one_hot_state}')
        sid_rob = self.vals2sid(nxtS = nxt_s[0])
        sid_hum = self.vals2sid(nxtS = nxt_s[1])
        one_hot_rob_s = self.get_one_hot(sid_rob, self.nSAgent)
        one_hot_hum_s = self.get_one_hot(sid_hum, self.nSAgent)
        # print("\nStep count Right before prev obs update: ", self._step_count)
        self.set_prev_obsv(0, sid_rob)
        self.set_prev_obsv(1, sid_hum)
        # print(f'Step {self._step_count}: Valid state and action Reward: {self.reward}')

        if self._step_count >= self._max_episode_steps:
            self._agent_dones = True

        if self.reward < self.step_cost:
            self._agent_dones = True
        # if self.reward != 0:
        #     self._agent_dones = True

        if self.steps_beyond_done is None and self._agent_dones:
            self.steps_beyond_done = 0
        elif self.steps_beyond_done is not None:
            if self.steps_beyond_done == 0:
                logger.warning(
                    f"Step {self._step_count}: You are calling 'step()' even though this environment has already returned all(dones) = True for "
                    "all agents. You should always call 'reset()' once you receive 'all(dones) = True' -- any further"
                    " steps are undefined behavior.")
            self.steps_beyond_done += 1
            self.reward = 0

        if self.custom:
            # return np.concatenate((one_hot_rob_s, one_hot_hum_s), axis=0), self.reward, self._agent_dones, {}
            return one_hot_state, self.reward, self._agent_dones, {}
        else:
            return [one_hot_rob_s, one_hot_hum_s], self.reward, self._agent_dones, {}

    def sid2vals(self, s):
        '''
        @brief - Given state id, this func converts it to the 3 variable values. 
        '''
        sid = s
        onionloc = int(mod(sid, self.nOnionLoc))
        sid = (sid - onionloc)/self.nOnionLoc
        eefloc = int(mod(sid, self.nEEFLoc))
        sid = (sid - eefloc)/self.nEEFLoc
        predic = int(mod(sid, self.nPredict))
        return onionloc, eefloc, predic

    def vals2sid(self, nxtS):
        '''
        @brief - Given the 3 variable values making up a state, this converts it into state id 
        '''
        ol = nxtS[0]
        eefl = nxtS[1]
        pred = nxtS[2]
        return (ol + self.nOnionLoc * (eefl + self.nEEFLoc * pred))

    def isValidAction(self, onionLoc, eefLoc, pred, action):
        '''
        @brief - For each state there are a few invalid actions, returns only valid actions.
        '''
        assert action <= 5, 'Unavailable action. Check if action is within num actions'
        if action == 0: # Noop, this can be done from anywhere.
            return True
        elif action == 1:   # Detect
            if pred == 0 or onionLoc == 0:  # Only when you don't know about the onion
                return True
            else: return False
        elif action == 2:   # Pick
            if onionLoc == 1 and eefLoc != 1:   # As long as onion is on conv and eef is not
                return True
            else: return False
        elif action == 3 or action == 4 or action == 5:   # Inspect # Placeonconv # Placeinbin
            if pred != 0 and onionLoc != 0: # Pred and onion loc both are known. 
                if onionLoc == eefLoc and eefLoc != 1:    # Onion is in hand and hand isn't on conv
                    return True
                else: return False
            else: return False
        else: 
            logger.error(f"Step {self._step_count}: Trying an impossible action are we? Better luck next time!")
            return False
        
    def findNxtState(self, onionLoc, eefLoc, pred, a):
        ''' 
        @brief - Returns the valid nextstates. Currently it's deterministic transition, could be made stochastic.
                This function assumes that you're doing the action from the appropriate current state.
                eg: If (onionloc - unknown, eefloc - onconv, pred - unknown), that's still a valid
                current state but an inappropriate state to perform inspect action and you shouldn't
                be able to transition into the next state induced by inspect. (Thanks @YikangGui for catching this.)
                Therefore inappropriate actions are filtered out by getValidActions method now. 

        Onionloc: {0: 'Unknown', 1: 'OnConveyor', 2: 'InFront', 3: 'AtHome'}
        eefLoc = {0: 'InBin', 1: 'OnConveyor', 2: 'InFront', 3: 'AtHome'}
        Predictions = {0:  'Unknown', 1: 'Bad', 2: 'Good}
        Actions: {0: 'Noop', 1: 'Detect', 2: 'Pick', 3: 'Inspect', 4: 'PlaceOnConveyor', 5: 'PlaceInBin}
        '''
        if a == 0:
            ''' Noop '''
            return [onionLoc, eefLoc, pred].copy()
        elif a == 1:
            ''' Detect '''
            prob = [0.8, 0.2]
            n_states = [[1, eefLoc, 2], [1, eefLoc, 1]]
            choice_index = np.random.choice(len(n_states), p=prob)
            return n_states[choice_index].copy()
        elif a == 2:
            ''' Pick '''
            return [3, 3, pred]
        elif a == 3:
            ''' Inspect '''
            if pred == 1:
                prob = [0.1, 0.9]
            else:
                prob = [0.8, 0.2]
            n_states = [[2, 2, 2], [2, 2, 1]]
            choice_index = np.random.choice(len(n_states), p=prob)
            return n_states[choice_index].copy()
        elif a == 4:
            ''' PlaceOnConv '''
            return [0, 1, 0]
        elif a == 5:
            ''' PlaceInBin '''
            return [0, 0, 0]

    def isValidState(self, onionLoc, eefLoc, pred):
        '''
        @brief - Checks if a given state is valid or not.

        '''
        if (onionLoc == 2 and eefLoc != 2) or (onionLoc == 3 and eefLoc != 3) or \
            (onionLoc == 0 and pred != 0) or (onionLoc != 0 and pred == 0):
            return False
        return True

    def getKeyFromValue(self, my_dict, val):
        for key, value in my_dict.items():
            if val == value:
                return key
        return "key doesn't exist"


AGENT_MEANING = {
    0: 'Robot',
    1: 'Human'
}

ACTION_MEANING = {
    0: 'Noop',
    1: 'Detect',
    2: 'Pick',
    3: 'Inspect',
    4: 'PlaceOnConveyor',
    5: 'PlaceinBin'
}

ONIONLOC = {
    0: 'Unknown',
    1: 'OnConveyor',
    2: 'InFront',
    3: 'AtHome'
}
EEFLOC = {
    0: 'InBin',
    1: 'OnConveyor',
    2: 'InFront',
    3: 'AtHome'
}

PREDICTIONS = {
    0: 'Unknown',
    1: 'Bad',
    2: 'Good'
}
