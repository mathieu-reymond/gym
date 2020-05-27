from gym.core import Env
from gym.spaces.box import Box
import numpy as np


class DamEnv(Env):
    """ A Water reservoir environment.
        The agent executes a continuous action, corresponding to the amount of water
        released by the dam.
        There are up to 4 rewards:
         - cost due to excess level wrt a flooding threshold (upstream)
         - deficit in the water supply wrt the water demand
         - deficit in hydroelectric supply wrt hydroelectric demand
         - cost due to excess level wrt a flooding threshold (downstream)

         Code from:
         https://gitlab.ai.vub.ac.be/mreymond/dam
         Ported from:
         https://github.com/sparisi/mips
    """

    S = 1.0 # Reservoir surface
    W_IRR = 50. # Water demand
    H_FLO_U = 50. # Flooding threshold (upstream, i.e. height of dam)
    S_MIN_REL = 100. # Release threshold (i.e. max capacity)
    DAM_INFLOW_MEAN = 40. # Random inflow (e.g. rain)
    DAM_INFLOW_STD = 10.
    Q_MEF = 0.
    GAMMA_H2O = 1000. # water density
    W_HYD = 4.36 # Hydroelectric demand
    Q_FLO_D = 30. # Flooding threshold (downstream, i.e. releasing too much water)
    ETA = 1. # Turbine efficiency
    G = 9.81 # Gravity

    utopia = {2: [-.5, -9], 3: [-.5, -9, -0.0001], 4: [-0.5, -9, -0.001, -9]}
    antiutopia = {2: [-2.5, -11], 3: [-65, -12, -0.7], 4: [-65, -12, -0.7, -12]}

    s_init = [9.6855361e+01, 
              5.8046026e+01, 
              1.1615767e+02, 
              2.0164311e+01, 
              7.9191000e+01, 
              1.4013098e+02, 
              1.3101816e+02,
              4.4351321e+01,
              1.3185943e+01,
              7.3508622e+01,]
    
    def __init__(self, nO=2, penalize=False):
        self.observation_space = Box(low=0, high=np.inf, shape=(1,))
        self.action_space = Box(low=0, high=np.inf, shape=(1,))

        self.nO = nO
        self.penalize = penalize

        low = -np.ones(nO)*np.inf # DamEnv.antiutopia[nO]
        high = np.zeros(nO) # DamEnv.utopia[nO]
        self.reward_space = Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    def reset(self):
        if not self.penalize:
            state = np.random.choice(DamEnv.s_init, size=1)
        else:
            state = np.random.randint(0, 160, size=1)
        
        self.state = state
        return self.state

    def step(self, action):

        # bound the action
        actionLB = np.clip(self.state - DamEnv.S_MIN_REL, 0, None)
        actionUB = self.state

        # Penalty proportional to the violation
        bounded_action = np.clip(action, actionLB, actionUB)
        penalty = -self.penalize*np.abs(bounded_action - action)

        # transition dynamic
        action = bounded_action
        dam_inflow = np.random.normal(DamEnv.DAM_INFLOW_MEAN, DamEnv.DAM_INFLOW_STD, len(self.state))
        # small chance dam_inflow < 0
        n_state = np.clip(self.state + dam_inflow - action, 0, None)

        # cost due to excess level wrt a flooding threshold (upstream)
        r0 = -np.clip(n_state/DamEnv.S - DamEnv.H_FLO_U, 0, None) + penalty
        # deficit in the water supply wrt the water demand
        r1 = -np.clip(DamEnv.W_IRR - action, 0, None) + penalty
        
        q = np.clip(action - DamEnv.Q_MEF, 0, None)
        p_hyd = DamEnv.ETA * DamEnv.G * DamEnv.GAMMA_H2O * n_state / DamEnv.S * q / 3.6e6

        # deficit in hydroelectric supply wrt hydroelectric demand
        r2 = -np.clip(DamEnv.W_HYD - p_hyd, 0, None) + penalty
        # cost due to excess level wrt a flooding threshold (downstream)
        r3 = -np.clip(action - DamEnv.Q_FLO_D, 0, None) + penalty

        reward = np.array([r0, r1, r2, r3], dtype=np.float32)[:self.nO].flatten()

        self.state = n_state
        return n_state, reward, False, {}
