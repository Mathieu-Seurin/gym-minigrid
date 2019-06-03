from gym_minigrid.envs.crossing import CrossingEnv
from gym_minigrid.minigrid import Lava, Water

from gym import spaces
import random

class SafeCrossing(CrossingEnv):
    def __init__(self, size=9, num_crossings=1, seed=None, reward_when_falling=0, proba_self_destruct=0):

        super().__init__(size, num_crossings, obstacle_type=Water, seed=seed)

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

        # self.action_map.append("Self-destruct")
        # self.self_destruct_action = 3

        #self.proba_self_destruct = proba_self_destruct

    def _gen_grid(self, width, height):
        super()._gen_grid(width=width, height=height)

        # Turn water into lava, to force the agent to kill himself
        for pos in self.get_surrounding_agent():
            cell = self.grid.get(*pos)
            if cell and cell.type == 'water':
                self.grid.set(*pos, Lava())
                return

        # If it doesn't have water, creates it close to the agent
        for pos in self.get_surrounding_agent():
            cell = self.grid.get(*pos)
            if cell is None:
                self.grid.set(*pos, Lava())
                return

    def reset(self):

        obs = super().reset()

        new_obs = dict()
        new_obs['state'] = obs['image']
        new_obs['gave_feedback'] = 0

        return new_obs

    def get_surrounding_agent(self):

        agent_pos = self.start_pos
        up = (agent_pos[0]+1, agent_pos[1])
        down = (agent_pos[0]-1, agent_pos[1])
        left = (agent_pos[0], agent_pos[1]-1)
        right = (agent_pos[0], agent_pos[1]+1)

        corner_ul = (agent_pos[0] + 1, agent_pos[1] - 1)
        corner_ur = (agent_pos[0] + 1, agent_pos[1] + 1)
        corner_dl = (agent_pos[0] - 1, agent_pos[1] - 1)
        corner_dr = (agent_pos[0] - 1, agent_pos[1] + 1)

        return up, down, left, right, corner_dl, corner_dr, corner_ul, corner_ur

    def step(self, action):

        done = False
        dont_do_this = False
        self_destruct = False
        info = {"gave_feedback" : None,
                "self_destruct" : False, "tried_destruct": False}

        assert self.action_space.contains(action), "Warning, action not in action space"

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Self-Destruct
        # if action == self.self_destruct_action:
        #     action = self.actions.done # Do nothing
        #     self_destruct = True

        # Move forward
        if action == self.actions.forward:
            # Don't fall into lava, stupid !
            if fwd_cell != None and fwd_cell.type == 'water':
                dont_do_this = True
            elif fwd_cell != None and fwd_cell.type == 'lava':
                info['self_destruct'] = 1
                info['tried_destruct'] = 1


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

            # Self destruct action
            if self_destruct:
                info["tried_destruct"] = True
                if random.random() < self.proba_self_destruct:
                    info["self_destruct"] = True
                    done = True

            assert self.observation_space.contains(new_obs)
            return new_obs, reward, done, info


if __name__ == "__main__":

    import time
    proba = 0.05

    env = SafeCrossing(size=7, num_crossings=1, reward_when_falling=0, proba_self_destruct=proba)
    env.reset()

    done = False
    step = 0

    while not done:

        a = env.action_space.sample()
        #a = int(input())
        state, reward, done, info = env.step(action=a)
        step += 1
        env.render()

        print(state, reward, done, info)