from gym_minigrid.envs.crossing import CrossingEnv
from gym_minigrid.minigrid import Lava

from gym import spaces

class SafeCrossing(CrossingEnv):
    def __init__(self, size=9, num_crossings=1, seed=None, reward_when_falling=0):
        super().__init__(size, num_crossings, obstacle_type=Lava, seed=seed)

        observation_space = {}

        observation_space['state'] = self.observation_space.spaces['image']
        observation_space['gave_feedback'] = spaces.Discrete(2)

        self.observation_space = spaces.Dict(observation_space)
        self.reward_when_falling = reward_when_falling

        # Reduce number of action so the problem is easier
        NUM_ACTIONS = 3

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.action_map = list(map(lambda x: str(x)[8:], self.actions))
        self.action_map = self.action_map[:NUM_ACTIONS]

    def reset(self):
        obs = super().reset()

        new_obs = dict()
        new_obs['state'] = obs['image']
        new_obs['gave_feedback'] = 0

        return new_obs

    def step(self, action):

        done = False
        dont_do_this = False
        info = {}

        assert self.action_space.contains(action), "Warning, action not in action space"

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward
        if action == self.actions.forward:

            # Don't fall into lava, stupid !
            if fwd_cell != None and fwd_cell.type == 'lava':
                dont_do_this = True

        if dont_do_this:
            self.step_count += 1
            if self.step_count >= self.max_steps:
                done = True
            obs = self.gen_obs()

            new_obs = dict()
            new_obs['state'] = obs['image']
            new_obs['gave_feedback'] = 1
            info['gave_feedback'] = 1

            return new_obs, self.reward_when_falling, done, info

        else:

            obs, reward, done, _ = super().step(action)

            new_obs = dict()
            new_obs['state'] = obs['image']
            new_obs['gave_feedback'] = 0
            info['gave_feedback'] = 0
            assert self.observation_space.contains(new_obs)
            return new_obs, reward, done, info



