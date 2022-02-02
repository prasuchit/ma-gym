import copy
import logging
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

    def __init__(self, full_observable=True, max_steps=100):

        global ACTION_MEANING, ONIONLOC, EEFLOC, PREDICTIONS, AGENT_MEANING
        self.n_agents = len(AGENT_MEANING)
        self._max_steps = max_steps
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

        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(6) for _ in range(self.n_agents)])
        self._obs_high = np.ones(self.nSAgent)
        self._obs_low = np.zeros(self.nSAgent)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high)
                                                             for _ in range(self.n_agents)])
        self.reward = 0
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
            assert np.sum(self.start[i]) == 1, "Start state probabilities don't add up to 1!"

    def get_action_meanings(self, agent_i=None):
        '''
        @brief - Just a translator to make it human-readable.
        '''
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def get_reward(self, acts):
        '''
        @brief Provides joint reward for appropriate team behavior.
        '''

        o_loc_rob, eef_loc_rob, pred_rob = self.sid2vals(self.prev_obsv[0])
        o_loc_hum, eef_loc_hum, pred_hum = self.sid2vals(self.prev_obsv[1])
        act_rob = acts[0]
        act_hum = acts[1]
        ######## INDEPENDENT FEATURES ########
        # If either agent doesn't know its onion loc, and decide to do detect
        if (o_loc_rob == 0 and act_rob == 1) or (o_loc_hum == 0 and act_hum == 1):
            self.reward += 1
        # If either agent knows its onion pred, the onion is on conv and decides to pick
        elif (pred_rob != 0 and o_loc_rob == 1 and act_rob == 2) or (pred_hum != 0 and o_loc_hum == 1 and act_hum == 2):
            self.reward += 1
        # If either agent has a bad onion pred, onion has been picked and decides to place in bin
        elif (pred_rob == 1 and o_loc_rob == 3 and act_rob == 5) or (pred_hum == 1 and o_loc_hum == 3 and act_hum == 5):
            self.reward += 1
        # If either agent has a good onion pred, onion has been picked and decides to inspect
        elif (pred_rob == 2 and o_loc_rob == 3 and act_rob == 3) or (pred_hum == 2 and o_loc_hum == 3 and act_hum == 3):
            self.reward += 1
        # If either agent has a bad onion pred, onion has been picked and decides to place on conveyor
        elif (pred_rob == 1 and o_loc_rob == 3 and act_rob == 4) or (pred_hum == 1 and o_loc_hum == 3 and act_hum == 4):
            self.reward -= 1
        # If either agent has a good onion pred, onion has been picked and decides to place in bin
        elif (pred_rob == 2 and o_loc_rob == 3 and act_rob == 5) or (pred_hum == 2 and o_loc_hum == 3 and act_hum == 5):
            self.reward -= 1
        # If either agent has a bad onion pred, onion has been picked and decides to inspect
        elif (pred_rob == 1 and o_loc_rob == 3 and act_rob == 3) or (pred_hum == 1 and o_loc_hum == 3 and act_hum == 3):
            self.reward -= 1
        # If either agent has inspected and found a good onion pred, and decides to place on conv
        elif (pred_rob == 2 and o_loc_rob == 2 and act_rob == 4) or (pred_hum == 2 and o_loc_hum == 2 and act_hum == 4):
            self.reward += 1
        # If either agent has inspected and found a bad onion pred, and decides to place in bin
        elif (pred_rob == 1 and o_loc_rob == 2 and act_rob == 5) or (pred_hum == 1 and o_loc_hum == 2 and act_hum == 5):
            self.reward += 1

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
        return np.concatenate((self.get_one_hot(s_r, self.nSAgent), self.get_one_hot(s_h, self.nSAgent)), axis = 0)

    def reset(self):
        '''
        @brief - Just setting all params to defaults and returning a valid start obsv.
        '''
        self._step_count = 0
        self.reward = 0
        self._agent_dones = False
        self.steps_beyond_done = None
        return self.get_init_obs()

    def set_prev_obsv(self, agent_id, s_id):
        self.prev_obsv[agent_id] = copy.copy(s_id)

    def get_prev_obsv(self, agent_id):
        return self.prev_obsv[agent_id]

    def get_one_hot(self, state, state_size):
        '''
        @brief - Given an integer and the max limit, returns it in one hot array form.
        '''
        assert 0 <= state < state_size, 'Invalid state! State should be b/w (0, state_size-1)'
        return np.squeeze(np.eye(state_size)[np.array(state).reshape(-1)])

    def step(self, agents_action):
        '''
        @brief - Performs given actions and returns one_hot(joint next obsvs), reward and done
        '''
        # agents_action = (agents_action // self.n_agents,    # Currently PPO returns 1 action mapped to all agents.
        #                  agents_action % self.n_agents)     # Here we're splitting it up to each agent action.
        assert len(agents_action) == self.n_agents, 'Num actions != num agents.'

        self._step_count += 1
        # print("Current step count is: ", self._step_count)
        sleep(0.1)
        nxt_s = {}
        for agent_i, action in enumerate(agents_action):
            o_loc, eef_loc, pred = self.sid2vals(self.prev_obsv[agent_i])
            if self.isValidState(o_loc, eef_loc, pred):
                nxt_s[agent_i] = self.findNxtState(
                    o_loc, eef_loc, pred, action)
            else:
                logger.error("Not a valid current state, ending episode!")
                self._agent_dones = True
                return self.reset(), self.reward, self._agent_dones

        self.get_reward(agents_action)
        sid_rob = self.vals2sid(nxtS = nxt_s[0])
        sid_hum = self.vals2sid(nxtS = nxt_s[1])
        one_hot_rob_s = self.get_one_hot(sid_rob, self.nSAgent)
        one_hot_hum_s = self.get_one_hot(sid_hum, self.nSAgent)
        self.set_prev_obsv(0, sid_rob)
        self.set_prev_obsv(1, sid_hum)

        if self._step_count >= self._max_steps:
            self._agent_dones = True

        if self.steps_beyond_done is None and self._agent_dones:
            self.steps_beyond_done = 0
        elif self.steps_beyond_done is not None:
            if self.steps_beyond_done == 0:
                logger.warning(
                    "You are calling 'step()' even though this environment has already returned all(dones) = True for "
                    "all agents. You should always call 'reset()' once you receive 'all(dones) = True' -- any further"
                    " steps are undefined behavior.")
            self.steps_beyond_done += 1
            self.reward = 0

        return np.concatenate((one_hot_rob_s, one_hot_hum_s), axis=0), self.reward, self._agent_dones

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

    def getValidActions(self, onionLoc, eefLoc, pred):
        ''' 
        @brief - For each state there are a few invalid actions, returns only valid actions.
        '''
        # I will come back to this if need be.
        pass

    def findNxtState(self, onionLoc, eefLoc, pred, a):
        ''' 
        @brief - For each state there are a few invalid nextstates, returns only valid nextstates.

        Onionloc: {0: 'Unknown', 1: 'OnConveyor', 2: 'InFront', 3: 'Picked/AtHome'}
        eefLoc = {0: 'OnConveyor', 1: 'InFront', 2: 'Picked/AtHome', 3:  'InBin'}
        Predictions = {0:  'Unknown', 1: 'Bad', 2: 'Good}
        Actions: {0: 'Noop', 1: 'Detect', 2: 'Pick', 3: 'Inspect', 4: 'PlaceOnConveyor', 5: 'PlaceInBin}
        '''
        if a == 0:
            ''' Noop '''
            return [onionLoc, eefLoc, pred]
        elif a == 1:
            ''' Detect '''
            n_states = [[1, eefLoc, 1], [1, eefLoc, 2]]
            choice_index = np.random.choice(len(n_states))
            return n_states[choice_index]
        elif a == 2:
            ''' Pick '''
            return [3, 2, pred]
        elif a == 3:
            ''' Inspect '''
            n_states = [[2, 1, pred], [2, 1, int(not pred)]]
            choice_index = np.random.choice(len(n_states))
            return n_states[choice_index]
        elif a == 4:
            ''' PlaceOnConv '''
            return [0, 0, 0]
        elif a == 5:
            ''' PlaceInBin '''
            return [0, 3, 0]

    def isValidState(self, onionLoc, eefLoc, pred):
        '''
        @brief - Checks if a given state is valid or not.

        '''
        if (onionLoc == 2 and eefLoc != 1) or (onionLoc == 3 and eefLoc != 2) or (onionLoc == 0 and pred != 0):
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
    3: 'Picked/AtHome'
}
EEFLOC = {
    0: 'OnConveyor',
    1: 'InFront',
    2: 'Picked/AtHome',
    3: 'InBin'
}

PREDICTIONS = {
    0: 'Unknown',
    1: 'Bad',
    2: 'Good'
}
