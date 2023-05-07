import copy
import logging
import random
from enum import *
from multiprocessing.sharedctypes import Value
from operator import mod
from time import sleep, time

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

# from ..utils.action_space import MultiAgentActionSpace
# from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)

AGENT_MEANING = {
    0: 'Robot',
    1: 'Human'
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

ACTION_MEANING = {
    0: 'Noop',
    1: 'Detect',
    2: 'Pick',
    3: 'Inspect',
    4: 'PlaceOnConveyor',
    5: 'PlaceinBin'
}

class DecHuRoSorting(gym.Env):
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

    def __init__(self, full_observable=False, max_steps=100):

        global ACTION_MEANING, ONIONLOC, EEFLOC, PREDICTIONS, AGENT_MEANING
        self.name = 'DecHuRoSorting'
        self.n_agents = len(AGENT_MEANING)
        self._max_episode_steps = max_steps
        self._step_count = None
        self.full_observable = full_observable
        assert not self.full_observable, "Decentralized agents can't have a fully observable global state! \
                                            Try HuRoSorting-v0 env if you're looking for a Multiagent MDP."
        self.nOnionLoc = len(ONIONLOC)
        self.nEEFLoc = len(EEFLOC)
        self.nPredict = len(PREDICTIONS)
        self.nInteract = 2
        self.nSAgent = self.nOnionLoc*self.nEEFLoc*self.nPredict*self.nInteract
        self.nAAgent = len(ACTION_MEANING)
        self.nSGlobal = (self.nSAgent)**self.n_agents
        self.nAGlobal = self.nAAgent**self.n_agents
        self.start = np.zeros((self.n_agents, self.nSAgent))
        self.prev_obsv = [None]*self.n_agents
        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.nAAgent) for _ in range(self.n_agents)])
        self._obs_high = np.ones(self.nOnionLoc+self.nEEFLoc+self.nPredict+self.nInteract)  
        self._obs_low = np.zeros(self.nOnionLoc+self.nEEFLoc+self.nPredict+self.nInteract) 
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high)
                                                             for _ in range(self.n_agents)])
        self.step_cost = 0.0
        self.reward = self.step_cost
        self._full_obs = None
        self._agent_dones = None
        self.steps_beyond_done = None
        self.verbose = False
        self.seed()

    def get_reward(self, acts):
        '''
        @brief Provides joint reward for appropriate team behavior.
        '''
        [o_loc_rob, eef_loc_rob, pred_rob, inter_rob] = self.sid2vals_interact(self.prev_obsv[0])
        [o_loc_hum, eef_loc_hum, pred_hum, inter_hum] = self.sid2vals_interact(self.prev_obsv[1])
        act_rob = acts[0]
        act_hum = acts[1]
        ######## INDEPENDENT FEATURES ########

        ##################### Robot #########################
        # Bad onion place in bin
        if pred_rob == 1 and act_rob == 5:
            self.reward += 1
        # Good onion place on conv
        elif pred_rob == 2 and act_rob == 4:
            self.reward += 1
        # Currently picked, find good, inspect
        elif o_loc_rob == 3 and pred_rob == 2 and act_rob == 3:
            self.reward += 1 
        # Bad onion place on conv
        elif pred_rob == 1 and act_rob == 4:
            self.reward -= 1
        # Good onion place in bin
        elif pred_rob == 2 and act_rob == 5:
            self.reward -= 1

        ##################### Human #########################
        # Bad onion place in bin
        if pred_hum == 1 and act_hum == 5:
            self.reward += 1
        # Good onion place on conv
        elif pred_hum == 2 and act_hum == 4:
            self.reward += 1
        # Currently picked, find good, inspect
        elif o_loc_hum == 3 and pred_hum == 2 and act_hum == 3:
            self.reward += 1
        # Bad onion place on conv
        elif pred_hum == 1 and act_hum == 4:
            self.reward -= 1
        # Good onion place in bin
        elif pred_hum == 2 and act_hum == 5:
            self.reward -= 1

        ######## DEPENDENT FEATURES ########

        # # Both do detect simultaneously
        # if (act_hum == act_rob == 1):
        #     self.reward -= 3

        # # Both pick simultaneously
        # if (act_hum == act_rob == 2):
        #     self.reward -= 3

        # # Both placeonconv simultaneously
        # if (act_hum == act_rob == 4):
        #     self.reward -= 3

        # If robot encounters an interaction and doesn't stop.
        if inter_rob == 1 and act_rob != 0:
            self.reward -= 3
            
    def render(self):
        self.verbose = True


    def reset(self, fixed_init=False):
        '''
        @brief - Just setting all params to defaults and returning a valid start obsv.
        '''
        self._step_count = 0
        self.reward = self.step_cost

        self._agent_dones = False
        self.steps_beyond_done = None

        return self.get_init_obs(fixed_init)

    ################ THIS IS CURRENTLY NOT USED ###############################
    # def failure_reset(self, fixed_init = False):
    #     random.seed(time())
    #     self._step_count = 0
    #     self.reward = self.step_cost
    #     self._agent_dones = False
    #     self.steps_beyond_done = None
    #     if fixed_init:
    #         state = [[0,3,0],[0,3,0]]
    #     else:
    #         state = random.choice([[[2,2,2],[2,2,2]],
    #                 [[3,3,2],[3,3,2]],
    #                 [[1,random.choice([0,2,3]),random.choice([1,2])],
    #                 [1,random.choice([0,2,3]),random.choice([1,2])]],
    #                 [[0,random.choice([0,2,3]),0],[0,random.choice([0,2,3]),0]]])
    #     self.set_prev_obsv(0, self.vals2sid(state[0]))
    #     self.set_prev_obsv(1, self.vals2sid(state[1]))
    #     return self.check_interaction(self.get_global_onehot(state))
    #############################################################################
    
    def check_interaction(self, s_r, s_h):
        interaction = 0
        [oloc_r, eefloc_r, pred_r, inter_r] = self.sid2vals_interact(s_r)
        [oloc_h, eefloc_h, pred_h, inter_h] = self.sid2vals_interact(s_h)
        if oloc_r == oloc_h == pred_r == pred_h == 2:   # Both oloc infront, pred good  - Placeonconv
            interaction = 1
        if oloc_r == oloc_h == pred_r == pred_h == 0:   # Both oloc and pred unknown    - Detect
            interaction = 1
        if oloc_r == oloc_h == 1 and (pred_r != 0 and pred_h != 0):  # Both oloc onconv, pred known - Pick
            interaction = 1
        robot_state = self.vals2sid_interact([oloc_r, eefloc_r, pred_r, interaction])
        human_state = self.vals2sid_interact([oloc_h, eefloc_h, pred_h, interaction])
        return robot_state, human_state

    def step(self, agents_action, verbose=None):
        '''
        @brief - Performs given actions and returns one_hot(joint next obsvs), reward and done
        '''
        if verbose != None:
            self.verbose = verbose
            
        agents_action = np.squeeze(agents_action)
        assert len(agents_action) == self.n_agents, 'Num actions != num agents.'
        self._step_count += 1
        self.reward = self.step_cost

        if self.verbose:
            [o_loc_0, eef_loc_0, pred_0, inter_0] = self.sid2vals_interact(self.prev_obsv[0])
            [o_loc_1, eef_loc_1, pred_1, inter_1] = self.sid2vals_interact(self.prev_obsv[1])
            print(f'Step {self._step_count}: Agent 0 state: {self.get_state_meanings(o_loc_0, eef_loc_0, pred_0, inter_0)} | Agent 1 state: {self.get_state_meanings(o_loc_1, eef_loc_1, pred_1, inter_1)}')
            print(f'Step {self._step_count}: Agent 0 action: {self.get_action_meanings(agents_action[0])} | Agent 1 action: {self.get_action_meanings(agents_action[1])}\n')

        nxt_s = {}
        for agent_i, action in enumerate(agents_action):
            [o_loc, eef_loc, pred, inter] = self.sid2vals_interact(self.prev_obsv[agent_i])
            if self.isValidState(o_loc, eef_loc, pred, inter):
                if self.isValidAction(o_loc, eef_loc, pred, inter, action):
                    nxt_s[agent_i] = self.findNxtState(o_loc, eef_loc, pred, inter, action)
                else:
                    if self.verbose:
                        logger.error(f"Step {self._step_count}: Invalid action: {self.get_action_meanings(action)}, in current state: {self.get_state_meanings(o_loc, eef_loc, pred, inter)}, agent {agent_i} can't transition anywhere else with this. Staying put and ending episode!")
                    self._agent_dones = True

                    ''' Sending all invalid actions to an impossible sink state'''

                    one_hot_state = [self.get_invalid_state()] * self.n_agents
                   
                    return one_hot_state, [self.reward] * self.n_agents, [self._agent_dones] * self.n_agents, {}
            else:
                if self.verbose:
                    logger.error(f"Step {self._step_count}: Invalid current state {self.get_state_meanings(o_loc, eef_loc, pred, inter)} for agent {agent_i}, ending episode!")
                self._agent_dones = True
                raise ValueError

        self.get_reward(agents_action)

        sid_rob = self.vals2sid_interact(sVals = nxt_s[0])
        sid_hum = self.vals2sid_interact(sVals = nxt_s[1])
        sid_rob, sid_hum = self.check_interaction(sid_rob, sid_hum) # Check if the next state we've reached is still an interactive state.
        self.set_prev_obsv(0, sid_rob)
        self.set_prev_obsv(1, sid_hum)

        one_hot_state = self.get_global_onehot([self.sid2vals_interact(sid_rob), self.sid2vals_interact(sid_hum)])

        if self._step_count >= self._max_episode_steps:
            self._agent_dones = True

        if self.reward < self.step_cost:
            self._agent_dones = True

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

        return one_hot_state, [self.reward] * self.n_agents, [self._agent_dones] * self.n_agents, {}

    def get_global_onehot(self, X):
        '''
        @brief: Returns a global one hot state using local states.
        '''
        one_hots = []
        for _, [onionloc, eefloc, pred, inter] in enumerate(X):
            onion_loc = self.get_one_hot(onionloc, self.nOnionLoc)
            eef_loc = self.get_one_hot(eefloc, self.nEEFLoc)
            prediction = self.get_one_hot(pred, self.nPredict)
            interaction = self.get_one_hot(inter, self.nInteract)
            one_hots.append(np.concatenate([onion_loc, eef_loc, prediction, interaction]))
        # print("Global one hot: ", one_hots)
        return one_hots

    def get_invalid_state(self):
        return np.concatenate([np.ones(self.nOnionLoc), np.ones(self.nEEFLoc), np.ones(self.nPredict), np.ones(self.nInteract)])  # last 1 is for interaction

    def get_init_obs(self, fixed_init=False):
        '''
        @brief - Samples from the start state distrib and returns a joint one hot obsv.
        '''
        if fixed_init:
            s_r = s_h = self.vals2sid([3,3,1,0])    # Athome, Athome, Bad, False
        else:
            self.update_start()
            s_r, s_h = self.sample_start()

        s_r, s_h = self.check_interaction(s_r, s_h)

        self.set_prev_obsv(0, s_r)
        self.set_prev_obsv(1, s_h)

        [onion_rob, eef_rob, pred_rob, inter_rob] = self.sid2vals_interact(s_r)

        [onion_hum, eef_hum, pred_hum, inter_hum] = self.sid2vals_interact(s_h)

        return self.get_global_onehot([[onion_rob, eef_rob, pred_rob, inter_rob], [onion_hum, eef_hum, pred_hum, inter_hum]])

    def sample_start(self):
        random.seed(time())
        sample_r = random.choices(np.arange(self.nSAgent), weights=np.reshape(self.start[0], (self.nSAgent)), k=1)[0]
        sample_h = random.choices(np.arange(self.nSAgent), weights=np.reshape(self.start[1], (self.nSAgent)), k=1)[0]
        return sample_r, sample_h

    def update_start(self):
        '''
        @brief - Sets the initial start state distrib. Currently, it's uniform b/w all valid states.
        '''
        for i in range(self.n_agents):
            for j in range(self.nSAgent):
                [o_l, eef_loc, pred, inter] = self.sid2vals_interact(j)
                if self.isValidStartState(o_l, eef_loc, pred, inter):
                    self.start[i][j] = 1
            self.start[i][:] = self.start[i][:] / \
                np.count_nonzero(self.start[i][:])
            assert np.sum(self.start[i]) == 1, "Start state distb doesn't add up to 1!"

    def isValidStartState(self, onionLoc, eefLoc, pred, inter):
        '''
        @brief - Checks if a given state is a valid start state or not.

        '''
        if (not self.isValidState(onionLoc, eefLoc, pred, inter)) or (eefLoc == 1):  # Can't start with eef on conv
            return False
        return True

    def get_action_meanings(self, action):
        '''
        @brief - Just a translator to make it human-readable.
        '''
        return ACTION_MEANING[action]
    
    def get_state_meanings(self, o_loc, eef_loc, pred, inter):
        '''
        @brief - Just a translator to make it human-readable.
        '''
        return ONIONLOC[o_loc], EEFLOC[eef_loc], PREDICTIONS[pred], bool(inter)

    def set_prev_obsv(self, agent_id, s_id):
        self.prev_obsv[agent_id] = copy.copy(s_id)

    def get_prev_obsv(self, agent_id):
        return self.prev_obsv[agent_id]

    def get_one_hot(self, x, max_size):
        '''
        @brief - Given an integer and the max limit, returns it in one hot array form.
        '''
        assert 0 <= x < max_size, 'Invalid value! x should be b/w (0, max_size-1)'
        return np.squeeze(np.eye(max_size)[np.array(x).reshape(-1)])

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
    
    def sid2vals_interact(self, s):
        '''
        @brief - Given state id, this func converts it to the 4 variable values. 
        '''
        sid = s
        onionloc = int(mod(sid, self.nOnionLoc))
        sid = (sid - onionloc)/self.nOnionLoc
        eefloc = int(mod(sid, self.nEEFLoc))
        sid = (sid - eefloc)/self.nEEFLoc
        predic = int(mod(sid, self.nPredict))
        sid = (sid - predic)/self.nPredict
        interact = int(mod(sid, self.nInteract))
        return [onionloc, eefloc, predic, interact]

    def vals2sid(self, sVals):
        '''
        @brief - Given the 3 variable values making up a state, this converts it into state id 
        '''
        ol = sVals[0]
        eefl = sVals[1]
        pred = sVals[2]
        return (ol + self.nOnionLoc * (eefl + self.nEEFLoc * pred))
    
    def vals2sid_interact(self, sVals):
        '''
        @brief - Given the 4 variable values making up a state, this converts it into state id 
        '''
        ol = sVals[0]
        eefl = sVals[1]
        pred = sVals[2]
        interact = sVals[3]
        return (ol + self.nOnionLoc * (eefl + self.nEEFLoc * (pred + self.nPredict * interact)))
    
    def vals2sGlobal(self, oloc_r, eefloc_r, pred_r, interact_r, oloc_h, eefloc_h, pred_h, interact_h):
        '''
        @brief - Given the variable values making up a global state, this converts it into global state id 
        '''
        return (oloc_r + self.nOnionLoc * (eefloc_r + self.nEEFLoc * (pred_r + self.nPredict *(interact_r + self.nInteract *(oloc_h + self.nOnionLoc * (eefloc_h + self.nEEFLoc * (pred_h + self.nPredict * interact_h)))))))

    def vals2aGlobal(self, a_r, a_h):
        '''
        @brief - Given the individual agent actions, this converts it into action id 
        '''
        return a_r + self.nAAgent * a_h

    def sGlobal2vals(self, s_global):
        '''
        @brief - Given the global state id, this converts it into the individual state variable values
        '''
        s_g = s_global
        oloc_r = int(mod(s_g, self.nOnionLoc))
        s_g = (s_g - oloc_r)/self.nOnionLoc
        eefloc_r = int(mod(s_g, self.nEEFLoc))
        s_g = (s_g - eefloc_r)/self.nEEFLoc
        pred_r = int(mod(s_g, self.nPredict))
        s_g = (s_g - pred_r)/self.nPredict
        interact_r = int(mod(s_g, self.nInteract))
        s_g = (s_g - interact_r)/self.nInteract
        oloc_h = int(mod(s_g, self.nOnionLoc))
        s_g = (s_g - oloc_h)/self.nOnionLoc
        eefloc_h = int(mod(s_g, self.nEEFLoc))
        s_g = (s_g - eefloc_h)/self.nEEFLoc
        pred_h = int(mod(s_g, self.nPredict))
        s_g = (s_g - pred_h)/self.nPredict
        interact_h = int(mod(s_g, self.nInteract))
        return oloc_r, eefloc_r, pred_r, interact_r, oloc_h, eefloc_h, pred_h, interact_h

    def aGlobal2vals(self, a_global):
        '''
        @brief - Given the global action, this converts it into individual action ids
        '''
        a_g = a_global
        a_r = int(mod(a_g, self.nAAgent))
        a_g = (a_g - a_r)/self.nAAgent
        a_h = int(mod(a_g, self.nAAgent))
        return a_r, a_h

    def isValidAction(self, onionLoc, eefLoc, pred, inter, action):
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
        
    def findNxtState(self, onionLoc, eefLoc, pred, inter, a):
        ''' 
        @brief - Returns the valid nextstates. 
        NOTE: @TBD make transition more stochastic.
        This function assumes that you're doing the action from the appropriate current state.
        eg: If (onionloc - unknown, eefloc - onconv, pred - unknown), that's still a valid
        current state but an inappropriate state to perform inspect action and you shouldn't
        be able to transition into the next state induced by inspect. (Thanks @YikangGui for catching this.)
        Therefore inappropriate actions are filtered out by getValidActions method now. 

        We don't mess with the interaction variable here, we return it as it is and let 
        checkInteraction() method determine if this state is an interaction state or not 
        because we don't have access to both agents' states here.

        Onionloc: {0: 'Unknown', 1: 'OnConveyor', 2: 'InFront', 3: 'AtHome'}
        eefLoc = {0: 'InBin', 1: 'OnConveyor', 2: 'InFront', 3: 'AtHome'}
        Predictions = {0:  'Unknown', 1: 'Bad', 2: 'Good}
        Actions: {0: 'Noop', 1: 'Detect', 2: 'Pick', 3: 'Inspect', 4: 'PlaceOnConveyor', 5: 'PlaceInBin}
        '''
        random.seed(time())
        if a == 0:
            ''' Noop '''
            return [onionLoc, eefLoc, pred, inter].copy()
        elif a == 1:
            ''' Detect '''
            '''NOTE: While doing detect the eef has to come back home.
                    Detect is done after placing somewhere and
                    if it stays at bin or on conv after detect,
                    that takes the transition to an invalid state.'''
            prob = [0.4, 0.6]
            n_states = [[1, 3, 1, inter], [1, 3, 2, inter]]
            choice = random.choices(n_states, weights=prob, k=1)[0]
            return choice
        elif a == 2:
            ''' Pick '''
            return [3, 3, pred, inter]
        elif a == 3:
            ''' Inspect '''
            if pred == 1:   # If initial pred is bad, it'll change to good very rarely 
                prob = [0.9, 0.1]
            else:   # If initial pred is good, it'll change more often
                prob = [0.3, 0.7]
            n_states = [[2, 2, 1, inter], [2, 2, 2, inter]]
            choice = random.choices(n_states, weights=prob, k=1)[0]
            return choice
        elif a == 4:
            ''' PlaceOnConv '''
            return [0, 1, 0, inter]
        elif a == 5:
            ''' PlaceInBin '''
            return [0, 0, 0, inter]

    def isValidState(self, onionLoc, eefLoc, pred, inter):
        '''
        @brief - Checks if a given state is valid or not.

        Interaction has nothing to do with a local state being valid or not.
        We just have the variable here because it's part of the local state.
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