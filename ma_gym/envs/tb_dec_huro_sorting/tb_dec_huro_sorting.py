import copy
import logging
import random
from enum import *
from operator import mod
from time import sleep, time

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)

AGENT_MEANING = {
    0: 'Robot',
    1: 'Human'
}

AGENT_TYPE = {
    0: 'Non-Fatigued',
    1: 'Fatigued'
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
    1: 'Detect_any',
    2: 'Detect_good',
    3: 'Detect_bad',
    4: 'Pick',
    5: 'Inspect',
    6: 'PlaceOnConveyor',
    7: 'PlaceinBin'
}

class TBDecHuRoSorting(gym.Env):
    """
    This environment is a slightly modified version of sorting env in this paper: Dec-AIRL(.pdf).
    This is a multi-agent sparse interactions Dec-MDP with 2 agents - a Human and a Robot. The Human agent is always given preference 
    while sorting and allowed to choose if there's a conflict. The Robot performs an independent sort while accounting for the Human's 
    local state and action. In order to learn this behavior, the Robot observes two Humans do the sort as a team where one of them 
    assume the role of the Robot.
    ------------------------------------------------------------------------------------------------------------------------
    Global state S - (s_rob, s_hum) x (type_rob, type_hum)
    Global action A - (a_rob, a_hum)
    Transitions T = Pr(S' | S, a_rob, a_hum)
    Joint Reward R(S,A) - Common reward that both agents get.
    Boolean variable eta - 1 if S is interactive state else 0.
    R(S,A) = eta*R_int + (1-eta)*R_non_int ; where R_int and R_non_int are reward for interaction and non-interaction cases.
    ------------------------------------------------------------------------------------------------------------------------
    State and action space are the same for both agents. 'agent' subscript below could mean either robot/human. 
    s_agent - (Onion_location, End_eff_loc, Onion_prediction)
    a_agent - (Noop, Detect_any, Detect_good, Pick, Inspect, PlaceOnConveyor, PlaceInBin)
    ------------------------------------------------------------------------------------------------------------------------
    Onion_location - Describes where the onion in focus currently is - (Unknown, OnConveyor, AtHome, InFront).
    End_eff_loc - Describes where the end effector currently is - (OnConveyor, AtHome, InFront, AtBin).
    Prediction - Provides the classification of the onion in focus - (Unknown, Good, Bad).
    Interaction - Boolean variable to say if agents are in interaction or not.
    NOTE: Onion location turns unknown when it's successfully placed on conv or in bin; 
    Until detect is done, both Prediction and Onion_location remain unknown.
    -------------------------------------------------------------------------------------------------------------------------
    Detect_any - Uses a classifier NN and CV techniques to find location and class of any onion - (Onion_location, Initial_Prediction)
    Detect_good - Same as Detect_any but only chooses a good onion as onion in focus.
    Detect_bad - Same as Detect_any but only chooses a bad onion as onion in focus.
    Pick - Dips down, grabs the onion in focus and comes back up - (Onion_location: AtHome, End_eff_loc: AtHome)
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

        global ACTION_MEANING, ONIONLOC, EEFLOC, PREDICTIONS, AGENT_MEANING, AGENT_TYPE
        self.name = 'TBDecHuRoSorting'
        self.n_agents = len(AGENT_MEANING)
        self._max_episode_steps = max_steps
        self._step_count = None
        self.verbose = False
        self.full_observable = full_observable
        assert not self.full_observable, "This is a Type-Based Decentralized MDP with local full observability! \
                                            Try HuRoSorting-v0 env if you're looking for fully observable multi-agent MDP."
        self.nOnionLoc = len(ONIONLOC)
        self.nEEFLoc = len(EEFLOC)
        self.nPredict = len(PREDICTIONS)
        self.nAgent_type = len(AGENT_TYPE)
        self.nInteract = 2
        ''' NOTE: The physical state include own type. But the observation space includes the own and other agent(s) types. '''
        self.nSAgent = self.nOnionLoc*self.nEEFLoc*self.nPredict*self.nInteract*self.nAgent_type
        self.nAAgent = len(ACTION_MEANING)
        self.nSGlobal = (self.nSAgent)**self.n_agents
        self.nAGlobal = self.nAAgent**self.n_agents
        self.start = np.zeros((self.n_agents, self.nSAgent))
        self.prev_obsv = [None]*self.n_agents
        self.prev_agent_type = [0]*self.n_agents
        self.true_label = [0]*self.n_agents
        self.num_onions_sorted_by_agent = [0]*self.n_agents
        self.onion_in_focus_status = ['']*self.n_agents
        random.seed(time())
        # self.onions_batch = [random.choice([1, 2]) for _ in range(100)] # 1 - Blemished onion, 2 - Unblemished onion.
        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.nAAgent) for _ in range(self.n_agents)])
        # NOTE: In obs_high and obs_low, one is self-type the other is other agent(s)' type that comes from belief update and not from step function.
        self._obs_high = np.ones(self.nOnionLoc+self.nEEFLoc+self.nPredict+self.nInteract+(self.nAgent_type * self.n_agents))
        self._obs_low = np.zeros(self.nOnionLoc+self.nEEFLoc+self.nPredict+self.nInteract+(self.nAgent_type * self.n_agents)) 
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high)
                                                            for _ in range(self.n_agents)])
        self.step_cost = 0.0
        self.reward = self.step_cost
        self._full_obs = None
        self._agent_dones = None
        self.steps_beyond_done = None
        self.seed()

    def _get_reward(self, acts):
        '''
        @brief Provides joint reward for appropriate team behavior.
        '''
        [_, _, _, inter_rob] = self.sid2vals_interact(self.prev_obsv[0])
        [_, _, _, _] = self.sid2vals_interact(self.prev_obsv[1])
        
        act_rob = acts[0]
        act_hum = acts[1]
        
        true_pred_rob = self.true_label[0]
        true_pred_hum = self.true_label[1]        
        
        rob_type = self.prev_agent_type[0]
        hum_type = self.prev_agent_type[1]
        
        
        ######## INDEPENDENT FEATURES ########

        ##################### Robot #########################
        if rob_type == 0:
            
            # Bad onion place in bin
            if true_pred_rob == 1 and act_rob == 5:
                self.reward += 1
            # Good onion place on conv
            elif true_pred_rob == 2 and act_rob == 4:
                self.reward += 1
                
            # # Currently picked, find good, inspect
            # elif o_loc_rob == 3 and pred_rob == 2 and act_rob == 3:
            #     self.reward += 1 
            
            # Bad onion place on conv
            elif true_pred_rob == 1 and act_rob == 4:
                self.reward -= 1
            # Good onion place in bin
            elif true_pred_rob == 2 and act_rob == 5:
                self.reward -= 1
        else:
            raise ValueError(f'Robot cannot be {AGENT_TYPE[rob_type]}.')

        ##################### Human #########################
        if hum_type == 0:   # Unfatigued
            # Bad onion place in bin
            if true_pred_hum == 1 and act_hum == 5:
                self.reward += 1
            # Good onion place on conv
            elif true_pred_hum == 2 and act_hum == 4:
                self.reward += 1
            # Bad onion place on conv
            elif true_pred_hum == 1 and act_hum == 4:
                self.reward -= 1
            # Good onion place in bin
            elif true_pred_hum == 2 and act_hum == 5:
                self.reward -= 1
        else:       # Fatigued
            # If good onion is chosen by fatigued human
            if true_pred_hum == 2:
                self.reward -= 10
            else:
                # Bad onion place in bin
                if true_pred_hum == 1 and act_hum == 5:
                    self.reward += 1
                # Bad onion place on conv
                elif true_pred_hum == 1 and act_hum == 4:
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

    def reset(self, fixed_init=False):
        '''
        @brief - Just setting all params to defaults and returning a valid start obsv.
        '''
        self._step_count = 0
        self.reward = self.step_cost

        self._agent_dones = False
        self.steps_beyond_done = None
        self.prev_agent_type = [0]*self.n_agents
        self.prev_obsv = [None]*self.n_agents
        self.true_label = [0]*self.n_agents        
        self.num_onions_sorted_by_agent = [0]*self.n_agents
        self.onion_in_focus_status = ['']*self.n_agents

        return self._get_init_obs(fixed_init)
    
    def render(self):
        self.verbose = True

    def _check_interaction(self, s_r, s_h):
        interaction = 0
        [oloc_r, eefloc_r, pred_r, _] = self.sid2vals_interact(s_r)
        [oloc_h, eefloc_h, pred_h, _] = self.sid2vals_interact(s_h)
        if oloc_r == oloc_h == pred_r == pred_h == 2:   # Both oloc infront, pred good  - Placeonconv
            interaction = 1
        if oloc_r == oloc_h == pred_r == pred_h == 0:   # Both oloc and pred unknown    - Detect
            interaction = 1
        if oloc_r == oloc_h == 1 and (pred_r != 0 and pred_h != 0):  # Both oloc onconv, pred known - Pick
            interaction = 1
        assert all(value != None for value in [oloc_r, eefloc_r, pred_r, interaction]), f"Some values in agent 0's state are None: {[oloc_r, eefloc_r, pred_r, interaction]}"
        assert all(value != None for value in [oloc_h, eefloc_h, pred_h, interaction]), f"Some values in agent 1's state are None: {[oloc_h, eefloc_h, pred_h, interaction]}"
        robot_state = self.vals2sid_interact([oloc_r, eefloc_r, pred_r, interaction])
        human_state = self.vals2sid_interact([oloc_h, eefloc_h, pred_h, interaction])
        return robot_state, human_state, bool(interaction)

    def step(self, agents_action):
        '''
        @brief - Performs given actions and returns one_hot (joint next obsvs), reward and done
        '''
        agents_action = np.squeeze(agents_action)
        assert len(agents_action) == self.n_agents, 'Num actions != num agents.'
        self._step_count += 1
        self.reward = self.step_cost

        # if len(self.onions_batch) == 0: # We're done sorting...
        #     self._agent_dones = True
        #     one_hot_state = self.get_global_onehot([self.sid2vals_interact(self.prev_obsv[i]) for i in self.n_agents])
        #     return one_hot_state, [self.reward] * self.n_agents, [self._agent_dones] * self.n_agents, {'info': 'Sorting complete', 'agent_types': self.prev_agent_type}
        verbose = True if self.verbose else False
        if verbose:
            [o_loc_0, eef_loc_0, pred_0, inter_0] = self.sid2vals_interact(self.prev_obsv[0])
            [o_loc_1, eef_loc_1, pred_1, inter_1] = self.sid2vals_interact(self.prev_obsv[1])
            agent_0, type_0 = self.get_agent_meanings(0, self.prev_agent_type[0])
            agent_1, type_1 = self.get_agent_meanings(1, self.prev_agent_type[1])
            print(f'Step {self._step_count}: Agent {agent_0} state: {self.get_state_meanings(o_loc_0, eef_loc_0, pred_0, inter_0)} | Agent {agent_1} state: {self.get_state_meanings(o_loc_1, eef_loc_1, pred_1, inter_1)}')
            print(f'Step {self._step_count}: Agent {agent_0} action: {self.get_action_meanings(agents_action[0])} | Agent {agent_1} action: {self.get_action_meanings(agents_action[1])}\n')
            print(f'Step {self._step_count}: Agent {agent_0} type: {type_0} | Agent {agent_1} type: {type_1}\n')

        nxt_s = {}
        for agent_i, action in enumerate(agents_action):
            [o_loc, eef_loc, pred, inter] = self.sid2vals_interact(self.prev_obsv[agent_i])
            if self._isValidState(o_loc, eef_loc, pred, inter):
                if self._isValidAction(o_loc, eef_loc, pred, inter, action, agent_i):
                    nxt_s[agent_i] = self._findNxtState(agent_id=agent_i, onionLoc=o_loc, eefLoc=eef_loc, pred=pred, inter=inter, a=action)
                    self._setNxtType(agent_i)
                else:
                    if verbose:
                        logger.error(f"Step {self._step_count}: Invalid action: {self.get_action_meanings(action)}, in current state: {self.get_state_meanings(o_loc, eef_loc, pred, inter)}, {AGENT_MEANING[agent_i]} agent can't transition anywhere else with this. Staying put and ending episode!")
                    self._agent_dones = True

                    ''' Sending all invalid actions to an impossible sink state'''

                    one_hot_state = [self._get_invalid_state()] * self.n_agents

                    return one_hot_state, [self.reward] * self.n_agents, [self._agent_dones] * self.n_agents, {'info': 'Invalid action reached a sink state.', 'agent_types': self.prev_agent_type}
            else:
                if verbose:
                    logger.error(f"Step {self._step_count}: Invalid current state {self.get_state_meanings(o_loc, eef_loc, pred, inter)} for {AGENT_MEANING[agent_i]} agent, ending episode!")
                self._agent_dones = True
                raise ValueError(f"Invalid state: Agent {agent_i} state: {self.get_state_meanings(o_loc, eef_loc, pred, inter)}, action:{action}.\n Shouldn't be reaching an invalid state!")

        self._get_reward(agents_action)
        assert all(value != None for value in nxt_s[0]), f"Some values in agent 0's state are None: {nxt_s[0]}"
        assert all(value != None for value in nxt_s[1]), f"Some values in agent 1's state are None: {nxt_s[1]}"
        sid_rob = self.vals2sid_interact(sVals = nxt_s[0])
        sid_hum = self.vals2sid_interact(sVals = nxt_s[1])
        sid_rob, sid_hum, is_interactive = self._check_interaction(sid_rob, sid_hum) # Check if the next state we've reached is still an interactive state.
        self._set_prev_obsv(0, sid_rob)
        self._set_prev_obsv(1, sid_hum)

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

        return one_hot_state, [self.reward] * self.n_agents, [self._agent_dones] * self.n_agents, {'info': 'Valid action, valid next state.', 'agent_types': self.prev_agent_type, 'interactive': is_interactive}

    def get_global_onehot(self, X):
        '''
        @brief: Returns a global one hot state using local states.
        '''
        one_hots = []
        for ag_id, [onionloc, eefloc, pred, inter] in enumerate(X):
            onion_loc = self.get_one_hot(onionloc, self.nOnionLoc)
            eef_loc = self.get_one_hot(eefloc, self.nEEFLoc)
            prediction = self.get_one_hot(pred, self.nPredict)
            interaction = self.get_one_hot(inter, self.nInteract)
            my_type = self.get_one_hot(self.prev_agent_type[ag_id], self.nAgent_type)
            one_hots.append(np.concatenate([onion_loc, eef_loc, prediction, interaction, my_type]))
        # print("Global one hot: ", one_hots)
        return one_hots

    def _get_invalid_state(self):
        return np.concatenate([np.ones(self.nOnionLoc), np.ones(self.nEEFLoc), np.ones(self.nPredict), np.ones(self.nInteract), np.ones(self.nAgent_type)])

    def _get_init_obs(self, fixed_init=False):
        '''
        @brief - Samples from the start state distrib and returns a joint one hot obsv.
        '''
        if fixed_init:
            s_r = s_h = self.vals2sid([3,3,1,0])    # Athome, Athome, Bad, False
        else:
            self._update_start()
            s_r, s_h = self._sample_start()

        s_r, s_h, _ = self._check_interaction(s_r, s_h) # System should never start in an interactive state, so we can assume interaction = False

        self._set_prev_obsv(0, s_r)
        self._set_prev_obsv(1, s_h)

        [onion_rob, eef_rob, pred_rob, inter_rob] = self.sid2vals_interact(s_r)

        [onion_hum, eef_hum, pred_hum, inter_hum] = self.sid2vals_interact(s_h)

        return self.get_global_onehot([[onion_rob, eef_rob, pred_rob, inter_rob], [onion_hum, eef_hum, pred_hum, inter_hum]])

    def _sample_start(self):
        random.seed(time())
        sample_r = random.choices(np.arange(self.nSAgent), weights=np.reshape(self.start[0], (self.nSAgent)), k=1)[0]
        sample_h = random.choices(np.arange(self.nSAgent), weights=np.reshape(self.start[1], (self.nSAgent)), k=1)[0]
        return sample_r, sample_h

    def _update_start(self):
        '''
        @brief - Sets the initial start state distrib. Currently, it's uniform b/w all valid states.
        '''
        for i in range(self.n_agents):
            for j in range(self.nSAgent):
                [o_l, eef_loc, pred, inter] = self.sid2vals_interact(j)
                if self._isValidStartState(o_l, eef_loc, pred, inter):
                    self.start[i][j] = 1
            self.start[i][:] = self.start[i][:] / \
                np.count_nonzero(self.start[i][:])
            assert np.sum(self.start[i]) == 1, "Start state distb doesn't add up to 1!"

    def _isValidStartState(self, onionLoc, eefLoc, pred, inter):
        '''
        @brief - Checks if a given state is a valid start state or not.

        '''
        return bool(self._isValidState(onionLoc, eefLoc, pred, inter) and eefLoc != 1)
    
    def _setNxtType(self, agent_id):        
        if self.prev_agent_type[agent_id] == 1: # Fatigued agent cannot become unfatigued during the same episode.
            return
        elif agent_id == 1 and self.num_onions_sorted_by_agent[agent_id] > 2:
            self._set_prev_agent_type(agent_id, 1)
            
    def get_other_agents_types(self, curr_ag_id):
        '''
        @brief Given a particular agent id, returns a list of all the other agents' true type.
        '''
        return [
            self.prev_agent_type[ag_id]
            for ag_id in range(self.n_agents)
            if ag_id != curr_ag_id
        ]

    def _set_prev_obsv(self, agent_id, s_id):
        self.prev_obsv[agent_id] = copy.copy(s_id)
        
    def _set_prev_agent_type(self, agent_id, agent_type):
        self.prev_agent_type[agent_id] = copy.copy(agent_type)

    def get_prev_obsv(self, agent_id):
        return self.prev_obsv[agent_id]

    def _isValidAction(self, onionLoc, eefLoc, pred, inter, action, agent_id):
        '''
        @brief - For each state there are a few invalid actions, returns only valid actions.
        '''
        assert action <= self.nAAgent, 'Unavailable action. Check if action is within num actions'
        if action == 0: # Noop, this can be done from anywhere.
            return True
        elif action in [1,2,3]:   # Detect, Detect_good, Detect_bad
            return pred == 0 or onionLoc == 0
        elif action == 4:   # Pick
            return onionLoc == 1 and eefLoc != 1
        elif action in [5, 6, 7]:   # Inspect # Placeonconv # Placeinbin
            return pred != 0 and self.true_label[agent_id] != 0 and onionLoc != 0 and onionLoc == eefLoc and eefLoc != 1
        else:
            logger.error(f"Step {self._step_count}: Trying an impossible action are we? Better luck next time!")
            return False
        
    def _findNxtState(self, agent_id, onionLoc, eefLoc, pred, inter, a):
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
        Actions: {0: 'Noop', 1: 'Detect_any', 2: 'Detect_good', 3: 'Pick', 4: 'Inspect', 5: 'PlaceOnConveyor', 6: 'PlaceinBin'}
        '''
        random.seed(time())
        if a == 0:
            ''' Noop '''
            return [onionLoc, eefLoc, pred, inter].copy()
        elif a == 1:
            ''' Detect '''
            self.true_label[agent_id] = random.choices([1,2], weights=[0.5, 0.5], k=1)[0]
            pred = self.true_label[agent_id] if self.true_label[agent_id] == 1 else random.choices([1,2], weights=[0.5, 0.5], k=1)[0]
            if self.onion_in_focus_status[agent_id] in ['', 'Placed']:
                self.onion_in_focus_status[agent_id] = 'Chosen'
            return [1, 3, pred, inter]
        elif a == 2:
            ''' Detect_good '''
            self.true_label[agent_id] = random.choices([1,2], weights=[0.5, 0.5], k=1)[0]
            pred = 2    # We perceive it as good, but we don't know the true label until we inspect
            if self.onion_in_focus_status[agent_id] in ['', 'Placed']:
                self.onion_in_focus_status[agent_id] = 'Chosen'
            return [1, 3, pred, inter]
        elif a == 3:
            ''' Detect_bad '''
            self.true_label[agent_id] = 1
            if self.onion_in_focus_status[agent_id] in ['', 'Placed']:
                self.onion_in_focus_status[agent_id] = 'Chosen'
            return [1, 3, self.true_label[agent_id], inter]
        elif a == 4:
            ''' Pick '''
            if self.onion_in_focus_status[agent_id] == 'Chosen':
                self.onion_in_focus_status[agent_id] = 'Picked'
            return [3, 3, pred, inter]
        elif a == 5:
            ''' Inspect '''
            if self.onion_in_focus_status[agent_id] == 'Picked':
                self.onion_in_focus_status[agent_id] = 'Inspected'
            return [2, 2, self.true_label[agent_id], inter]
        elif a == 6:
            ''' PlaceOnConv '''
            if self.onion_in_focus_status[agent_id] in ['Picked', 'Inspected']:
                self.onion_in_focus_status[agent_id] = 'Placed'
                self.num_onions_sorted_by_agent[agent_id] += 1
            return [0, 1, 0, inter]
        elif a == 7:
            ''' PlaceInBin '''
            if self.onion_in_focus_status[agent_id] in ['Picked', 'Inspected']:
                self.onion_in_focus_status[agent_id] = 'Placed'
                self.num_onions_sorted_by_agent[agent_id] += 1
            return [0, 0, 0, inter]

    # def _get_detected_state(self, agent_id, onionLoc, eefLoc, pred, inter, action):

    #     '''NOTE: While doing detect the eef has to come back home.
    #         Detect is done after placing somewhere and
    #         if it stays at bin or on conv after detect,
    #         that takes the transition to an invalid state.'''
            
    #     # create a list of indices where the good onions are in the onions_batch
    #     indices = [i for i in range(len(self.onions_batch)) if self.onions_batch[i] == 2]

    #     # check if the list of indices is empty or Detect action is done.
    #     if not indices or action == 1:
    #         # choose a random index
    #         index = random.randrange(len(self.onions_batch))
    #         # remove the element at the chosen index
    #     else:   # if Detect_good action is done
    #         # choose a random index
    #         index = random.choice(indices)

    #     # remove the element at the chosen index
    #     self.true_label[agent_id] = self.onions_batch.pop(index)
    #     # if it is a blemished onion, we return the true label, else detected label is wrong 50% of the time 
    #     detected_label = self.true_label[agent_id] if self.true_label[agent_id] == 1 else random.choice([1,2])
    #     return [1, 3, detected_label, inter]    # onconv, athome, prediction, interaction

    def _isValidState(self, onionLoc, eefLoc, pred, inter):
        '''
        @brief - Checks if a given state is valid or not.

        Interaction has nothing to do with a local state being valid or not.
        We just have the variable here because it's part of the local state.
        '''
        return (
            (onionLoc != 2 or eefLoc == 2)
            and (onionLoc != 3 or eefLoc == 3)
            and (onionLoc != 0 or pred == 0)
            and (onionLoc == 0 or pred != 0)
        )

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
    
    def get_agent_meanings(self, agent_id, ag_type):
        '''
        @brief - Just a translator to make it human-readable.
        '''
        return AGENT_MEANING[agent_id], AGENT_TYPE[ag_type]

    
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


    def getKeyFromValue(self, my_dict, val):
        return next(
            (key for key, value in my_dict.items() if val == value),
            "key doesn't exist",
        )