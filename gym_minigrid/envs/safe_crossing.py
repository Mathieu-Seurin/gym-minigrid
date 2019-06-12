from gym_minigrid.envs.crossing import CrossingEnv
from gym_minigrid.minigrid import Lava, Water

from gym import spaces
import random

class SafeCrossing(CrossingEnv):
    def __init__(self, size, reward_when_falling, feedback_when_wall_hit, proba_reset, use_lava=True, n_more_actions=0, num_crossings=1, seed=None):

        super().__init__(size, num_crossings, obstacle_type=Water, seed=seed)

        observation_space = {}

        observation_space['state'] = self.observation_space.spaces['image']
        observation_space['gave_feedback'] = spaces.Discrete(2)

        self.observation_space = spaces.Dict(observation_space)
        self.reward_when_falling = reward_when_falling

        # Reduce number of action so the problem is easier
        num_actions = 3

        self.action_map = list(map(lambda x: str(x)[8:], self.actions))
        self.action_map = self.action_map[:num_actions]

        if n_more_actions:
            num_actions += n_more_actions
            self.action_map.extend(["Useless action {}".format(i) for i in range(n_more_actions)])

        self.action_space = spaces.Discrete(num_actions)

        # Feedback when going in water AND wall too ?
        self.feedback_when_hit_wall = feedback_when_wall_hit
        self.proba_reset = proba_reset

        self.use_lava = use_lava


    def _gen_grid(self, width, height):
        super()._gen_grid(width=width, height=height)

        if self.use_lava:
            directions = self.get_surrounding_agent()

            # Turn water into lava, to force the agent to kill himself
            for pos in directions:
                cell = self.grid.get(*pos)
                if cell and cell.type == 'water':
                    self.grid.set(*pos, Lava())
                    return


            random.shuffle(directions)
            # If it doesn't have water, creates it close to the agent
            for pos in directions:
                cell = self.grid.get(*pos)

                assert pos != self.start_pos

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

        directions = [up, down, left, right, corner_dl, corner_dr, corner_ul, corner_ur]
        return directions

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

            # Don't hit wall, stupid !
            elif self.feedback_when_hit_wall and fwd_cell != None and fwd_cell.type == "wall":
                dont_do_this = True
                
            elif fwd_cell != None and fwd_cell.type == 'lava':
                info['self_destruct'] = 1
                info['tried_destruct'] = 1


        if dont_do_this:
            self.step_count += 1
            if self.step_count >= self.max_steps and self.proba_reset < random.random():
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


if __name__ == "__main__":

    import time
    from env_tools.wrapper import MinigridFrameStacker

    env = SafeCrossing(size=7, proba_reset=0, feedback_when_wall_hit=0, reward_when_falling=0, n_more_actions=10)
    env = MinigridFrameStacker(env, n_frameskip=3)

    save = set()

    for i in range(1000):

        env.reset()

        done = False
        step = 0
        sum_reward = 0

        while not done:

            #env.render()

            a = env.action_space.sample()
            #a = int(input())
            state, reward, done, info = env.step(action=a)

            sum_reward += reward

            step += 1

            #env.render()


            #print(state, reward, done, info)

        if sum_reward > 0:
            save.add((step, sum_reward))



    save = list(save)
    save.sort(key=lambda x:x[1])
    print(save)
