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

BASELINE = False

AGENT_MEANING = {
    0: 'Robot',
    1: 'Human'
}

AGENT_TYPE_HUM = {
    0: 'Unfatigued',
    1: 'Fatigued',
}
AGENT_TYPE_ROB = {
    0: 'Collaborative',
    1: 'Super-Collaborative',
}

INDICATIONS = {
    0: False,
    1: True    
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

ACTION_MEANING_HUM = {
    0: 'Noop',
    1: 'Detect',
    2: 'Detect_pick',
    3: 'Pick',
    4: 'Inspect',
    5: 'PlaceOnConveyor',
    6: 'PlaceinBin',
    7: 'Thumbs_up',
    8: 'Thumbs_down',
}

ACTION_MEANING_ROB = {
    0: 'Noop',
    1: 'Detect',
    2: 'Detect_pick',
    3: 'Pick',
    4: 'Inspect',
    5: 'PlaceOnConveyor',
    6: 'PlaceinBin',
    7: 'Speed_up',
    8: 'Slow_down',
}

class TBDecHuRoSorting(gym.Env):
    """
    This environment is a slightly modified version of sorting env in this paper: Dec-AIRL(.pdf).
    This is a multi-agent sparse interactions Dec-MDP with 2 agents - a Human and a Robot. The Human agent is always given preference 
    while sorting and allowed to choose if there's a conflict. The Robot performs an independent sort while accounting for the Human's 
    preferences. In order to learn this behavior, the Robot observes two Humans do the sort as a team where one of them 
    assume the role of the Robot.
    ------------------------------------------------------------------------------------------------------------------------
    Global state S - (s_rob, s_hum)
    Global action A - (a_rob, a_hum)
    Transitions T = Pr(S' | S, A)
    Joint Reward R(S,A) - Common reward that both agents get.
    The observation space defined within this gym env includes the belief state dimension (for easy implementation) for each agent,
    but this belief does not play into the MDP dynamics or reward and is only used by the policy.
    ------------------------------------------------------------------------------------------------------------------------
    State and action space are the same for both agents. 'agent' subscript below could mean either robot/human. 
    s_agent - (Onion_location, End_eff_loc, Onion_prediction, Interaction_flag, Self_type, Indication_flag)
    a_agent - (Noop, Detect_any, Detect_good, Pick, Inspect, PlaceOnConveyor, PlaceInBin)
    ------------------------------------------------------------------------------------------------------------------------
    Onion_location - Describes where the onion in focus currently is - (Unknown, OnConveyor, AtHome, InFront).
    End_eff_loc - Describes where the end effector currently is - (OnConveyor, AtHome, InFront, AtBin).
    Onion_prediction - Provides the classification of the onion in focus - (Unknown, Good, Bad).
    Interaction_flag - Boolean variable to say if agents are in interaction or not.
    Self_type - Variable holding the value of the agent's own type.
    Indication_flag - Informs whether the current type has been correctly informed to the other agent. 
    NOTE: Onion location turns unknown when it's successfully placed on conv or in bin; 
    Until detect is done, both Onion_location and Onion_prediction remain unknown.
    -------------------------------------------------------------------------------------------------------------------------
    Noop - No action is done here - Meant to be used by Robot when both agents want to detect/placeonconveyor at the same time.
    Detect - Uses a classifier NN and CV techniques to find location and class of any onion - (Onion_location, Initial_Prediction)
    Pick - Dips down, grabs the onion in focus and comes back up - (Onion_location: AtHome, End_eff_loc: AtHome)
    Detect_pick - A combined action of quick detection and picking to reduce number of timesteps per sort - meant for super-collaborative robot.
    Inspect - If Initial_Prediction is good, we need to inspect again to make sure, onion is shown close to camera - (Onion_location: InFront, End_eff_loc: InFront). 
    PlaceOnConveyor - If Prediction turns out to be good, it's placed back on the conveyor and liftup - (Onion_location: Unknown, End_eff_loc: OnConv).
    PlaceInBin - If Initial_Prediction is bad, the onion is directly placed in the bad bin - (Onion_location: Unknown, End_eff_loc: AtBin). 
    Human-specific:
        Thumbs_up - Human shows a thumbs up sign to the camera, which indicates to the robot (noisily) that he is unfatigued,
        Thumbs_down - Human shows a thumbs down sign to the camera, which indicates to the robot (noisily) that he is fatigued,
    Robot-specific:
        Speed_up - Robot increases its speed of movement indicating to the human that it is now in super-collaborative mode,
        Slow_down - Robot slows down its movement indicating to the human that it is in collaborative mode now.
    -------------------------------------------------------------------------------------------------------------------------
    Episode starts from one of the valid start states where eef is anywhere, onion is on conv and pred is unknown.
    Episode ends when one onion is successfully chosen, picked, inspected and placed somewhere.
    """

    """ 
    NOTE: TBD - Make transition stochastic.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, full_observable=False, max_steps=100):

        global ACTION_MEANING_HUM, ACTION_MEANING_ROB, ONIONLOC, EEFLOC, PREDICTIONS, \
            AGENT_MEANING, AGENT_TYPE_HUM, AGENT_TYPE_ROB, INDICATIONS
        self.name = 'TBDecHuRoSorting'
        self.n_agents = len(AGENT_MEANING)
        self._max_episode_steps = max_steps
        self._step_count = 0
        self.verbose = False
        self.full_observable = full_observable
        assert not self.full_observable, "This is a Type-Based Decentralized MDP with local full observability! \
                                            Try HuRoSorting-v0 env if you're looking for fully observable multi-agent MDP."
        self.nOnionLoc = len(ONIONLOC)
        self.nEEFLoc = len(EEFLOC)
        self.nPredict = len(PREDICTIONS)
        self.nAgent_type = len(AGENT_TYPE_HUM)
        self.nInteract = 2
        self.nIndicate = len(INDICATIONS)
        ''' NOTE: The physical state includes only self type. But the observation space includes self and other agent(s) types. '''
        self.nSAgent = self.nOnionLoc*self.nEEFLoc*self.nPredict*self.nInteract*self.nAgent_type*self.nIndicate
        self.nAAgent = len(ACTION_MEANING_HUM)
        self.nSGlobal = (self.nSAgent)**self.n_agents
        self.nAGlobal = self.nAAgent**self.n_agents
        self.start = np.zeros((self.n_agents, self.nSAgent))
        self.prev_obsv = [None]*self.n_agents
        self._agent_type = [0, 0]
        self.num_onions_sorted_by_agent = [0, 0]
        self.num_onions_sorted_total = 0
        self.indicated = {ag_id: 0 for ag_id in range(self.n_agents)}
        self.recovery_time_count = 0
        self.human_recovery_time = 5
        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.nAAgent) for _ in range(self.n_agents)])
        # NOTE: In obs_high and obs_low, one is self-type the other is other agent(s)' type that comes from belief update and not from step function.
        if not BASELINE:
            self._obs_high = np.ones(self.nOnionLoc+self.nEEFLoc+self.nPredict+self.nInteract+self.nAgent_type+self.nIndicate+self.nAgent_type)
            self._obs_low = np.zeros(self.nOnionLoc+self.nEEFLoc+self.nPredict+self.nInteract+self.nAgent_type+self.nIndicate+self.nAgent_type) 
        else:   
            self._obs_high = np.ones(self.nOnionLoc+self.nEEFLoc+self.nPredict+self.nInteract+self.nAgent_type+self.nIndicate)
            self._obs_low = np.zeros(self.nOnionLoc+self.nEEFLoc+self.nPredict+self.nInteract+self.nAgent_type+self.nIndicate) 
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high)
                                                            for _ in range(self.n_agents)])
        self.step_cost = -0.02
        self.reward = self.step_cost
        self.nxt_s = {}
        self._full_obs = None
        self._agent_dones = None
        self.steps_beyond_done = None
        self.max_onions_possible = 70
        self.seed()

    def _get_reward(self, acts):
        '''
        @brief Provides joint reward for appropriate team behavior.
        '''
        [o_loc_rob, _, pred_rob, inter_rob, type_r, _] = self.sid2vals(self.prev_obsv[0])
        [o_loc_hum, _, pred_hum, _, type_h, indicate_h] = self.sid2vals(self.prev_obsv[1])
        
        act_rob = acts[0]
        act_hum = acts[1]
        
        ######## INDEPENDENT FEATURES ########

        ##################### Robot #########################    
        # Bad onion place in bin
        if pred_rob == 1 and act_rob == 6:
            self.reward += 1
        # Good onion place on conv
        elif pred_rob == 2 and act_rob == 5:
            self.reward += 1            
        # Currently picked, find good, inspect
        elif o_loc_rob == 3 and pred_rob == 2 and act_rob == 4:
            self.reward += 1 
        # Bad onion place on conv
        elif pred_rob == 1 and act_rob == 5:
            self.reward -= 1
        # Good onion place in bin
        elif pred_rob == 2 and act_rob == 6:
            self.reward -= 1        
        
        ##################### Human #########################
        if type_h == 0:      # Unfatigued type
            # Bad onion place in bin
            if pred_hum == 1 and act_hum == 6:
                self.reward += 1
            # Good onion place on conv
            elif pred_hum == 2 and act_hum == 5:
                self.reward += 1
            # Currently picked, find good, inspect
            elif o_loc_hum == 3 and pred_hum == 2 and act_hum == 4:
                self.reward += 1 
            # Bad onion place on conv
            elif pred_hum == 1 and act_hum == 5:
                self.reward -= 1
            # Good onion place in bin
            elif pred_hum == 2 and act_hum == 6:
                self.reward -= 1        
        
        # ######## DEPENDENT FEATURES ########
        
        # if type_h == 1 and indicate_h == 1 and type_r != 1:
        #     self.reward -= 1
        # elif type_h == 0 and indicate_h == 1 and type_r != 0:
        #     self.reward -= 1

        # # If robot encounters an interaction and doesn't stop.
        # if inter_rob == 1 and act_rob != 0:
        #     self.reward -= 3

    def reset(self, fixed_init=False):
        '''
        @brief - Just setting all params to defaults and returning a valid start obsv.
        '''
        self._step_count = 0
        self.reward = self.step_cost
        self._agent_dones = False
        self.steps_beyond_done = None
        self._agent_type = [0, 0]
        self.prev_obsv = [None]*self.n_agents
        self.num_onions_sorted_by_agent = [0, 0]
        self.num_onions_sorted_total = 0
        self.recovery_time_count = 0
        self.indicated = {ag_id: 0 for ag_id in range(self.n_agents)}
        for i in range(self.n_agents):
            self.action_space[i].seed(random.seed(time()))
        return self._get_init_obs(fixed_init)
    
    def render(self):
        self.verbose = True

    def _check_interaction(self, s_r, s_h):
        interaction = 0
        [oloc_r, eefloc_r, pred_r, _, self_type_r, indicate_r] = self.sid2vals(s_r)
        [oloc_h, eefloc_h, pred_h, _, self_type_h, indicate_h] = self.sid2vals(s_h)
        if oloc_r == oloc_h == pred_r == pred_h == 2:   # Both oloc infront, pred good  - Placeonconv
            interaction = 1
        if oloc_r == oloc_h == pred_r == pred_h == 0:   # Both oloc and pred unknown    - Detect
            interaction = 1
        if oloc_r == oloc_h == 1 and (pred_r != 0 and pred_h != 0):  # Both oloc onconv, pred known - Pick
            interaction = 1
        assert all(value != None for value in [oloc_r, eefloc_r, pred_r, interaction, self_type_r, indicate_r]), f"Some values in agent 0's state are None: {[oloc_r, eefloc_r, pred_r, interaction, self_type_r, indicate_r]}"
        assert all(value != None for value in [oloc_h, eefloc_h, pred_h, interaction, self_type_h, indicate_h]), f"Some values in agent 1's state are None: {[oloc_h, eefloc_h, pred_h, interaction, self_type_h, indicate_h]}"
        robot_state = self.vals2sid([oloc_r, eefloc_r, pred_r, interaction, self_type_r, indicate_r])
        human_state = self.vals2sid([oloc_h, eefloc_h, pred_h, interaction, self_type_h, indicate_h])
        return robot_state, human_state, bool(interaction)

    def step(self, agents_action):
        '''
        @brief - Performs given actions and returns one_hot (joint next obsvs), reward and done
        '''
        agents_action = np.squeeze(agents_action)
        assert len(agents_action) == self.n_agents, 'Num actions != num agents.'
        self._step_count += 1
        self.reward = self.step_cost
        verbose = bool(self.verbose)
        if verbose:
            [o_loc_0, eef_loc_0, pred_0, inter_0, self_type_0, indicate_0] = self.sid2vals(self.prev_obsv[0])
            [o_loc_1, eef_loc_1, pred_1, inter_1, self_type_1, indicate_1] = self.sid2vals(self.prev_obsv[1])
            agent_0, type_0 = self.get_agent_meanings(0, self._agent_type[0])
            agent_1, type_1 = self.get_agent_meanings(1, self._agent_type[1])
            print("--"*25, "Start of Step function", "--"*25)
            print(f'Step {self._step_count}: Agent {agent_0} state: {self.get_state_meanings(o_loc_0, eef_loc_0, pred_0, inter_0, self_type_0, indicate_0, ag_id=0)} | Agent {agent_1} state: {self.get_state_meanings(o_loc_1, eef_loc_1, pred_1, inter_1, self_type_1, indicate_1, ag_id=1)}\n')
            print(f'Step {self._step_count}: Agent {agent_0} action: {self.get_action_meanings(agents_action[0], ag_id=0)} | Agent {agent_1} action: {self.get_action_meanings(agents_action[1], ag_id=1)}\n')
            print(f'Step {self._step_count}: Agent {agent_0} type: {type_0} | Agent {agent_1} type: {type_1}\n')
            if type_1 == 'Fatigued':
                print(f'Step {self._step_count}: Agent {agent_1} is fatigued! Recovering in {self.human_recovery_time - self.recovery_time_count} steps.\n')
                
        self.nxt_s = {}
        for agent_i, action in enumerate(agents_action):
            [o_loc, eef_loc, pred, inter, self_type, indicate] = self.sid2vals(self.prev_obsv[agent_i])
            if self._isValidState(o_loc, eef_loc, pred, inter, self_type, indicate):
                if self._isValidAction(o_loc, eef_loc, pred, inter, self_type, indicate, action, agent_i):
                    self.nxt_s[agent_i] = self._findNxtState(agent_id=agent_i, onionLoc=o_loc, eefLoc=eef_loc, pred=pred, inter=inter, indicate=indicate, ag_type=self_type, a=action)
                else:
                    if verbose:
                        print(f"Step {self._step_count}: Invalid action: {self.get_action_meanings(action, ag_id=agent_i)}, in current state: {self.get_state_meanings(o_loc, eef_loc, pred, inter, self_type, indicate, ag_id=agent_i)}, {AGENT_MEANING[agent_i]} agent can't transition anywhere else with this. Staying put and ending episode!")
                    self._agent_dones = True

                    ''' Sending all invalid actions to an impossible sink state'''

                    one_hot_state = [self._get_invalid_state()] * self.n_agents

                    return one_hot_state, [self.reward] * self.n_agents, [self._agent_dones] * self.n_agents, {'info': 'Invalid action reached a sink state.', 'agent_types': self._agent_type}
            else:
                if verbose:
                    print(f"Step {self._step_count}: Invalid current state {self.get_state_meanings(o_loc, eef_loc, pred, inter, self_type, indicate, ag_id=agent_i)} for {AGENT_MEANING[agent_i]} agent, ending episode!")
                self._agent_dones = True
                raise ValueError(f"Invalid state: Agent {agent_i} state: {self.get_state_meanings(o_loc, eef_loc, pred, inter, self_type, indicate, ag_id=agent_i)}, action:{self.get_action_meanings(action, agent_i)}.\n Shouldn't be reaching an invalid state!")

        self._setNxtType(1)     # Human type updates exogenously first and robot adjusts accordingly
        self._get_reward(agents_action)
        assert all(value != None for value in self.nxt_s[0]), f"Some values in agent 0's state are None: {self.nxt_s[0]}"
        assert all(value != None for value in self.nxt_s[1]), f"Some values in agent 1's state are None: {self.nxt_s[1]}"
        sid_rob = self.vals2sid(sVals = self.nxt_s[0])
        sid_hum = self.vals2sid(sVals = self.nxt_s[1])
        sid_rob, sid_hum, is_interactive = self._check_interaction(sid_rob, sid_hum) # Check if the next state we've reached is still an interactive state.
        [o_loc_r, eef_loc_r, pred_r, inter_r, self_type_r, indicate_r] = self.sid2vals(sid_rob) 
        [o_loc_h, eef_loc_h, pred_h, inter_h, self_type_h, indicate_h] = self.sid2vals(sid_hum)
        # Testing if internally changing the indicate var of robot helps learning.
        # Ideally the belief state should inform the robot to switch type, but somehow it isn't
        if self_type_h == 1 and indicate_h == 1 and self_type_r == 0:
            indicate_r = 0
        elif self_type_h == 0 and indicate_h == 1 and self_type_r == 1:
            indicate_r = 0
        sid_rob = self.vals2sid(sVals = [o_loc_r, eef_loc_r, pred_r, inter_r, self_type_r, indicate_r])
        sid_hum = self.vals2sid(sVals = [o_loc_h, eef_loc_h, pred_h, inter_h, self_type_h, indicate_h])
        self._set_prev_obsv(0, sid_rob)
        self._set_prev_obsv(1, sid_hum)
        one_hot_state = self.get_global_onehot([self.sid2vals(sid_rob), self.sid2vals(sid_hum)])

        if self._step_count >= self._max_episode_steps:
            self._agent_dones = True

        if self.steps_beyond_done is None and self._agent_dones:
            self.steps_beyond_done = 0
        elif self.steps_beyond_done is not None:
            if self.steps_beyond_done == 0:
                print(
                    f"Step {self._step_count}: You are calling 'step()' even though this environment has already returned all(dones) = True for "
                    "all agents. You should always call 'reset()' once you receive 'all(dones) = True' -- any further"
                    " steps are undefined behavior.")
            self.steps_beyond_done += 1
            self.reward = 0
            
        if verbose:
            agent_0, type_0 = self.get_agent_meanings(0, self._agent_type[0])
            agent_1, type_1 = self.get_agent_meanings(1, self._agent_type[1])
            print("After updating step...\n")
            print(f'Step {self._step_count}: Agent {agent_0} reward: {self.reward} | Agent {agent_1} reward: {self.reward}\n')
            print(f'Step {self._step_count}: Agent {agent_0} done: {self._agent_dones} | Agent {agent_1} done: {self._agent_dones}\n')        
            print("--"*25, "End of Step function", "--"*25)

        return one_hot_state, [self.reward] * self.n_agents, [self._agent_dones] * self.n_agents, {'info': 'Valid action, valid next state.', 'agent_types': self._agent_type, 'interactive': is_interactive}

    def get_global_onehot(self, X):
        '''
        @brief: Returns a global one hot state using local states.
        '''
        one_hots = []
        for _, [onionloc, eefloc, pred, inter, self_type, indicate] in enumerate(X):
            onion_loc = self.get_one_hot(onionloc, self.nOnionLoc)
            eef_loc = self.get_one_hot(eefloc, self.nEEFLoc)
            prediction = self.get_one_hot(pred, self.nPredict)
            interaction = self.get_one_hot(inter, self.nInteract)
            self_type = self.get_one_hot(self_type, self.nAgent_type)
            indicate = self.get_one_hot(indicate, self.nIndicate)
            one_hots.append(np.concatenate([onion_loc, eef_loc, prediction, interaction, self_type, indicate]))
        # print("Global one hot: ", one_hots)
        return one_hots

    def _get_init_obs(self, fixed_init=True):
        '''
        @brief - Samples from the start state distrib and returns a joint one hot obsv.
        '''
        if fixed_init:
            s_r = s_h = self.vals2sid([0,3,0,0,0,1])    # Unknown, Athome, Unknown, False, Default_type, Indicated
        else:
            self._update_start()
            s_r, s_h = self._sample_start()

        s_r, s_h, _ = self._check_interaction(s_r, s_h)
        
        self._set_prev_obsv(0, s_r)
        self._set_prev_obsv(1, s_h)

        [onion_rob, eef_rob, pred_rob, inter_rob, type_rob, indicate_rob] = self.sid2vals(s_r)

        [onion_hum, eef_hum, pred_hum, inter_hum, type_hum, indicate_hum] = self.sid2vals(s_h)
        
        self._set_agent_type(0, type_rob)
        self._set_agent_type(1, type_hum)
        self.indicated[0] = indicate_rob
        self.indicated[1] = indicate_hum
        

        return self.get_global_onehot([[onion_rob, eef_rob, pred_rob, inter_rob, type_rob, indicate_rob], [onion_hum, eef_hum, pred_hum, inter_hum, type_hum, indicate_hum]])

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
                [o_l, eef_loc, pred, inter, ag_type, indicate] = self.sid2vals(j)
                if self._isValidStartState(o_l, eef_loc, pred, inter, ag_type, indicate):
                    self.start[i][j] = 1
            self.start[i][:] = self.start[i][:] / \
                np.count_nonzero(self.start[i][:])
            assert np.sum(self.start[i]) == 1, "Start state distb doesn't add up to 1!"

    def _setNxtType(self, agent_id, a = None):    
        
        # For human, the type update happens exogenously    
        if agent_id == 1:   # Human
            # If human has sorted some onions
            onion_limit = np.random.randint(2,6)
            if self._agent_type[agent_id] == 0 and self.num_onions_sorted_by_agent[agent_id] >= onion_limit:
                # They are now fatigued
                self._set_agent_type(agent_id, 1)
                # Reset num onions sorted by unfatigued agent
                self.num_onions_sorted_by_agent[agent_id] = 0
                
            # If human is currently fatigued
            elif self._agent_type[agent_id] == 1:
                # Start tracking their recovery time
                self.recovery_time_count += 1
                # If it has been max recovery timesteps since fatigue
                if self.recovery_time_count >= self.human_recovery_time:
                    # They switch back to unfatigued now
                    self._set_agent_type(agent_id, 0)
                    # Reset num onions sorted by fatigued agent
                    self.num_onions_sorted_by_agent[agent_id] = 0
                    # Reset recovery time
                    self.recovery_time_count = 0
        
        # For robot, its indicative action induces type change
        elif agent_id == 0:    # Robot
            # If robot is in normal collab mode and act is speedup
            if self._agent_type[agent_id] == 0 and a == 7:
                # Robot becomes super-collaborative
                self._set_agent_type(agent_id, 1)
            # If robot is in super collab mode and act is slowdown
            elif self._agent_type[agent_id] == 1 and a == 8:
                self._set_agent_type(agent_id, 0)
    
    def get_other_agents_types_ground_truth(self, curr_ag_id):
        '''
        @brief Given a particular agent id, returns a list of all 
        the other agents' true type.
        '''
        return [
            self._agent_type[ag_id]
            for ag_id in range(self.n_agents)
            if ag_id != curr_ag_id
        ]    
            
    def get_other_agents_types_for_belief_training(self, curr_ag_id):
        '''
        @brief Given a particular agent id, returns a list of all the other agents' probable type.
        This method is only used to train the belief network to make it learn the belief distribution.
        This is not used during policy training or execution.
        '''
        # return [
        #     random.choices(np.arange(self.nAgent_type), weights=[0.95, 0.05], k=1)[0]
        #     for ag_id in range(self.n_agents) if ag_id != curr_ag_id
        # ]
        return [
            self._agent_type[ag_id]
            for ag_id in range(self.n_agents)
            if ag_id != curr_ag_id
        ]  
    
    def _findNxtState(self, agent_id, onionLoc, eefLoc, pred, inter, ag_type, indicate, a):
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
        EefLoc = {0: 'InBin', 1: 'OnConveyor', 2: 'InFront', 3: 'AtHome'}
        Predictions = {0:  'Unknown', 1: 'Bad', 2: 'Good}
        Interaction = {0: True, 1: False}
        Agent_Type = Robot {0: 'Collaborative', 1: 'Super Collaborative'} ; Human {0: 'Unfatigued', 1: 'Fatigued'}
        Indication = {0: False, 1: True}
        Actions_Rob: {0: 'Noop', 1: 'Detect', 2: 'Detect_pick', 3: 'Pick', 4: 'Inspect', 5: 'PlaceOnConveyor', 6: 'PlaceinBin', 7: 'Speed_up', 8: 'Slow_down'}
        Actions_Hum: {0: 'Noop', 1: 'Detect', 2: 'Detect_pick', 3: 'Pick', 4: 'Inspect', 5: 'PlaceOnConveyor', 6: 'PlaceinBin', 7: 'Thumbs_up', 8: 'Thumbs_down'}
        '''
        random.seed(time())     
        
        # Updating agent_type and indicate vars of the state
        if self._agent_type[agent_id] != ag_type:
            ag_type = self._agent_type[agent_id]
            indicate = 0            
        
        if a == 0:
            ''' Noop '''
            # Noop takes eef home to induce some change in the state so that the logic doesn't get stuck in a loop. 
            if eefLoc in [0, 1] and onionLoc == pred == 0:
                eefLoc = 3                               
        elif a == 1:
            ''' Detect '''
            pred = random.choices([1,2], weights=[0.5, 0.5], k=1)[0]
            onionLoc = 1
            eefLoc = 3
        elif a == 2:
            ''' Detect_pick '''
            pred = random.choices([1,2], weights=[0.5, 0.5], k=1)[0]
            onionLoc = 3
            eefLoc = 3
        elif a == 3:
            ''' Pick '''
            onionLoc = 3
            eefLoc = 3
        elif a == 4:
            ''' Inspect '''
            pred = random.choices([1,2], weights=[0.5, 0.5], k=1)[0] if pred == 2 else pred
            onionLoc = 2
            eefLoc = 2
        elif a == 5:
            ''' PlaceOnConv '''
            self.num_onions_sorted_by_agent[agent_id] += 1
            self.num_onions_sorted_total += 1
            onionLoc = 0
            eefLoc = 1
            pred = 0
        elif a == 6:
            ''' PlaceInBin '''
            self.num_onions_sorted_by_agent[agent_id] += 1
            self.num_onions_sorted_total += 1
            onionLoc = 0
            eefLoc = 0
            pred = 0
        elif a in [7,8]:
            ''' Speed_up/slow_down for robot or Thumbs_up/thumbs_down for human '''            
            if agent_id == 0:   # For robot, an indicative action changes its type
                self._setNxtType(agent_id, a)             
            indicate = 1
            
            
        ag_type = self._agent_type[agent_id]
        self.indicated[agent_id] = indicate
        
        return [onionLoc, eefLoc, pred, inter, ag_type, indicate].copy()

    def _isValidStartState(self, onionLoc, eefLoc, pred, inter, ag_type, indicate):
        '''
        @brief - Checks if a given state is a valid start state or not.
        '''
        # Has to be a valid state, eef not on conv, not interactive, 
        # ag_type not evolved yet, start with indicated true to avoid random indications
        # return bool(self._isValidState(onionLoc, eefLoc, pred, inter, ag_type, indicate) \
        #         and eefLoc != 1 and inter != 1 and ag_type != 1 and indicate != 0)
        return bool(self._isValidState(onionLoc, eefLoc, pred, inter, ag_type, indicate) \
                and eefLoc != 1 and indicate != 0)
        
    def _isValidState(self, onionLoc, eefLoc, pred, inter, self_type, indicate):
        '''
        @brief - Checks if a given state is valid or not.

        Interaction has nothing to do with a local state being valid or not.
        Same with self_type and indicate.
        We just have these variables here because they're part of the local state.
        '''
        # SIDE-NOTE for the future: De Morgan's law of Boolean Algebra lets us rewrite the
        # following conjunctions into disjunctions to yield the same effect as the following
        # condition. However, I'm using conjunctions since it's more readable. The disjunction
        # version of these conditions are given as: 
        # return ((onionLoc != 2 or eefLoc == 2) and (onionLoc != 3 or eefLoc == 3) and
        # (onionLoc != 0 or pred == 0) and (onionLoc == 0 or pred != 0)    
        
        return not (
            (onionLoc == 2 and eefLoc != 2) or   # Onion is in front, eef is not
            (onionLoc == 3 and eefLoc != 3) or   # Onion is at home, eef is not
            (onionLoc == 0 and pred != 0) or     # Onion loc is unknown but pred isn't
            (onionLoc != 0 and pred == 0)        # Onion loc is known but pred isn't
        )
    
    def _isValidAction(self, onionLoc, eefLoc, pred, inter, ag_type, indicate, action, agent_id):
        '''
        @brief - For each state there are a few invalid actions, returns only valid actions.
        Actions_Rob: {0: 'Noop', 1: 'Detect', 2: 'Detect_pick', 3: 'Pick', 4: 'Inspect', 5: 'PlaceOnConveyor', 6: 'PlaceinBin', 7: 'Speed_up', 8: 'Slow_down'}
        Actions_Hum: {0: 'Noop', 1: 'Detect', 2: 'Detect_pick', 3: 'Pick', 4: 'Inspect', 5: 'PlaceOnConveyor', 6: 'PlaceinBin', 7: 'Thumbs_up', 8: 'Thumbs_down'}
        '''
        assert action <= self.nAAgent, 'Unavailable action. Check if action is within num actions'
        
        # Fatigued, indicated human can't do any other action except NoOp
        if agent_id == 1 and ag_type == 1 and indicate == 1 and action != 0:
            return False     
        else:
            # All actions must have an indicate condition to enforce agent to indicate
            if action == 0:     # Noop, this can be done from anywhere.
                return indicate == 1 and ag_type != 0 and agent_id != 0
            elif action == 1:   # Detect
                return (pred == 0 or onionLoc == 0) and ag_type == 0 and indicate == 1
            elif action == 2:   # Detect_pick
                return (pred == 0 or onionLoc == 0) and (agent_id == 0 and ag_type == 1) and indicate == 1
            elif action == 3:   # Pick
                return onionLoc == 1 and eefLoc != 1 and indicate == 1
            elif action in [4, 5, 6]:   # Inspect # Placeonconv # Placeinbin
                return pred != 0 and onionLoc != 0 and onionLoc == eefLoc and eefLoc != 1 and indicate == 1
            elif action == 7:   # Speedup for rob; thumbs-up for human
                # NOTE: For human, this is just an indicative action to inform his updated type, whereas for robot this action 
                # changes its type. So for human the updated type has to be considered and for robot the type is yet to change.
                
                # return (ag_type == 0 and inter != 1 and indicate != 1) if agent_id == 0 else (ag_type == 0 and indicate != 1)
                return (ag_type == 0 and indicate != 1)
            elif action == 8:   # Slow-down for rob; thumbs-down for human 
                # NOTE: For human, this is just an indicative action to inform his updated type, whereas for robot this action 
                # changes its type. So for human the updated type has to be considered and for robot the type is yet to change.
                
                # return (ag_type == 1 and inter != 1 and indicate != 1) if agent_id == 0 else (ag_type == 1 and indicate != 1)
                return (ag_type == 1 and indicate != 1)
            else:
                print(f"Step {self._step_count}: Trying an impossible action are we? Better luck next time!")
                return False
    
    def _get_invalid_state(self):
        return np.concatenate([np.ones(self.nOnionLoc), np.ones(self.nEEFLoc), np.ones(self.nPredict), np.ones(self.nInteract), np.ones(self.nAgent_type), np.ones(self.nIndicate)])
    
    def _set_prev_obsv(self, agent_id, s_id):
        self.prev_obsv[agent_id] = copy.copy(s_id)
        
    def _set_agent_type(self, agent_id, agent_type):
        self._agent_type[agent_id] = copy.copy(agent_type)

    def get_prev_obsv(self, agent_id):
        return self.prev_obsv[agent_id]
    
    def get_action_meanings(self, action, ag_id=0):
        '''
        @brief - Just a translator to make it human-readable.
        '''
        return ACTION_MEANING_ROB[action] if ag_id == 0 else ACTION_MEANING_HUM[action]
    
    def get_state_meanings(self, o_loc, eef_loc, pred, inter, self_type, indicate, ag_id=0):
        '''
        @brief - Just a translator to make it human-readable.
        '''
        type_list = AGENT_TYPE_ROB if ag_id == 0 else AGENT_TYPE_HUM
        return f"Oloc: {ONIONLOC[o_loc]}", f"Eefloc: {EEFLOC[eef_loc]}", f"Pred: {PREDICTIONS[pred]}", \
            f"Interact: {bool(inter)}", f"Self-type: {type_list[self_type]}", f"Indicated: {INDICATIONS[indicate]}"
    
    def get_agent_meanings(self, agent_id, ag_type):
        '''
        @brief - Just a translator to make it human-readable.
        '''
        ag_type = AGENT_TYPE_ROB[ag_type] if agent_id == 0 else AGENT_TYPE_HUM[ag_type]
        return AGENT_MEANING[agent_id], ag_type
    
    def get_one_hot(self, x, max_size):
        '''
        @brief - Given an integer and the max limit, returns it in one hot array form.
        '''
        assert 0 <= x < max_size, 'Invalid value! x should be b/w (0, max_size-1)'
        return np.squeeze(np.eye(max_size)[np.array(x).reshape(-1)])

    def sid2vals(self, s):
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
        sid = (sid - interact)/self.nInteract
        my_type = int(mod(sid, self.nAgent_type))
        sid = (sid - my_type)/self.nAgent_type
        indicate = int(mod(sid, self.nIndicate))
        
        return [onionloc, eefloc, predic, interact, my_type, indicate]
    
    def vals2sid(self, sVals):
        '''
        @brief - Given the 4 variable values making up a state, this converts it into state id 
        '''
        ol = sVals[0]
        eefl = sVals[1]
        pred = sVals[2]
        interact = sVals[3]
        my_type = sVals[4]
        indicate = sVals[5]
        return (ol + self.nOnionLoc * (eefl + self.nEEFLoc * (pred + self.nPredict * \
            (interact + self.nInteract * (my_type + self.nAgent_type * (indicate))))))
    
    def vals2sGlobal(self, oloc_r, eefloc_r, pred_r, interact_r, type_r, indicate_r, \
                    oloc_h, eefloc_h, pred_h, interact_h, type_h, indicate_h):
        '''
        @brief - Given the variable values making up a global state, this converts it into global state id 
        '''
        return (oloc_r + self.nOnionLoc * \
                (eefloc_r + self.nEEFLoc * \
                (pred_r + self.nPredict * \
                (interact_r + self.nInteract * \
                (type_r + self.nAgent_type * \
                (indicate_r + self.nIndicate * \
                (oloc_h + self.nOnionLoc * \
                (eefloc_h + self.nEEFLoc * \
                (pred_h + self.nPredict * \
                (interact_h + self.nInteract * \
                (type_h + self.nAgent_type * \
                (indicate_h)
                )))))))))))

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
        type_r = int(mod(s_g, self.nAgent_type))
        s_g = (s_g - type_r)/self.nAgent_type
        indicate_r = int(mod(s_g, self.nIndicate))
        s_g = (s_g - indicate_r)/self.nIndicate  
        oloc_h = int(mod(s_g, self.nOnionLoc))
        s_g = (s_g - oloc_h)/self.nOnionLoc
        eefloc_h = int(mod(s_g, self.nEEFLoc))
        s_g = (s_g - eefloc_h)/self.nEEFLoc
        pred_h = int(mod(s_g, self.nPredict))
        s_g = (s_g - pred_h)/self.nPredict
        interact_h = int(mod(s_g, self.nInteract))
        s_g = (s_g - interact_h)/self.nInteract
        type_h = int(mod(s_g, self.nAgent_type))
        s_g = (s_g - type_h)/self.nAgent_type
        indicate_h = int(mod(s_g, self.nIndicate))
        return oloc_r, eefloc_r, pred_r, interact_r, type_r, indicate_r,\
                oloc_h, eefloc_h, pred_h, interact_h, type_h, indicate_h

    def vals2aGlobal(self, a_r, a_h):
        '''
        @brief - Given the individual agent actions, this converts it into action id 
        '''
        return a_r + self.nAAgent * a_h

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