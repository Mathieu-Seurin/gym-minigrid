from gym_minigrid.envs.crossing import CrossingEnv
from gym_minigrid.minigrid import Lava, Water, Floor, Wall


from gym import spaces
import random

from gym_minigrid.minigrid import COLORS, IDX_TO_COLOR
from itertools import product
from copy import deepcopy

class SafeCrossing(CrossingEnv):
    def __init__(self, size, reward_when_falling, feedback_when_wall_hit, proba_reset, n_zone=1, use_lava=False, n_more_actions=0, seed=None, good_zone_action_proba=0.85, bad_zone_action_proba=0.5, obstacle_type='wall'):

        self.obstacle_type = obstacle_type
        if obstacle_type == "wall":
            obstacle = Wall
            num_crossings = 1

        elif obstacle_type == "water":
            obstacle = Water
            num_crossings = 1

        elif obstacle_type == "none":
            obstacle = Water
            num_crossings = 0

        self.use_lava = use_lava
        self.n_zone = n_zone
        self.n_reset = 0

        super().__init__(size, num_crossings, obstacle_type=obstacle, seed=seed)

        observation_space = {}

        observation_space['state'] = self.observation_space.spaces['image']
        observation_space['gave_feedback'] = spaces.Discrete(2)
        observation_space['zone'] = spaces.Discrete(self.n_zone)

        self.observation_space = spaces.Dict(observation_space)
        self.reward_when_feedback = reward_when_falling

        # Reduce number of action so the problem is easier
        self.num_basic_actions = 3
        num_actions = 3

        self.action_map = list(map(lambda x: str(x)[8:], self.actions))
        self.action_map = self.action_map[:num_actions]
        self.action_to_zone = dict([(act,0) for act in range(self.num_basic_actions)])
        self.to_basic_action = dict([(act, act) for act in range(self.num_basic_actions)])

        self.good_zone_proba = 1
        self.current_zone_num = 0

        # For every zone possible, create a new set of action
        if self.n_zone > 1:
            new_action_map = []
            new_action_to_zone = dict()
            new_to_basic_action = dict()

            for zone in range(self.n_zone):
                new_action_map.extend([action + str(zone) for action in self.action_map])
                for action in range(num_actions) :
                    high_action = num_actions*zone+action
                    new_action_to_zone[high_action] = zone
                    new_to_basic_action[high_action] = high_action % self.num_basic_actions

            self.action_to_zone = new_action_to_zone
            self.action_map = new_action_map
            self.to_basic_action = new_to_basic_action

            num_actions = num_actions * self.n_zone

            self.good_zone_proba = good_zone_action_proba
            self.bad_zone_proba = bad_zone_action_proba

        if n_more_actions:
            num_actions += n_more_actions
            self.action_map.extend(["Useless action {}".format(i) for i in range(n_more_actions)])

        self.action_space = spaces.Discrete(num_actions)

        # Feedback when going in water AND wall too ?
        self.feedback_when_hit_wall = feedback_when_wall_hit
        self.proba_reset = proba_reset

    def change_floor_color(self, color='blue'):

        for x,y in product(range(self.grid.height), range(self.grid.width)):
            if self.grid.get(x,y) is None:
                floor = Floor(color=color)
                self.grid.set(x,y, floor)


    def _gen_grid(self, width, height):
        super()._gen_grid(width=width, height=height)

        if self.n_zone > 1:
            self.current_zone_num = self.n_reset % self.n_zone
            color = IDX_TO_COLOR[self.current_zone_num+1]
            self.change_floor_color(color=color)

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

                if cell is None or cell.type == 'floor':
                    self.grid.set(*pos, Lava())
                    return

    def reset(self):

        obs = super().reset()
        self.n_reset += 1

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
        give_feedback = False

        info = {"self_destruct" : False,
                "tried_destruct": False}

        assert self.action_space.contains(action), "Warning, action not in action space"

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)


        # if the zone is the correct one : execute good action with high proba
        # else : execute action with a lower proba

        good_zone = self.action_to_zone[action] == self.current_zone_num
        if good_zone:
            keep_action_proba = self.good_zone_proba
        else:
            keep_action_proba = self.bad_zone_proba
            give_feedback = True

        if random.random() > keep_action_proba:
            action = random.randint(0,self.num_basic_actions-1) # upper bound is included, so need to adjust
        else:
            action = self.to_basic_action[action]

        # Move forward
        if action == self.actions.forward:
            # Don't fall into lava, stupid !
            if fwd_cell != None and fwd_cell.type == 'water':
                dont_do_this = True
                give_feedback = True

            # Don't hit wall, stupid !
            elif self.feedback_when_hit_wall and fwd_cell != None and fwd_cell.type == "wall":
                dont_do_this = True
                give_feedback = True
                
            elif fwd_cell != None and fwd_cell.type == 'lava':
                info['self_destruct'] = 1
                info['tried_destruct'] = 1

        new_obs = dict()
        new_obs['gave_feedback'] = give_feedback
        new_obs['zone'] = self.current_zone_num

        info['gave_feedback'] = give_feedback

        if give_feedback:
            reward = self.reward_when_feedback
        else:
            reward = 0

        if dont_do_this:
            self.step_count += 1
            if self.step_count >= self.max_steps or random.random() < self.proba_reset:
                done = True

            obs = self.gen_obs()
            new_obs['state'] = obs['image']

            assert self.observation_space.contains(new_obs)
            return new_obs, reward, done, info

        else:
            obs, new_reward, done, _ = super().step(action)
            new_obs['state'] = obs['image']

            assert self.observation_space.contains(new_obs)
            return new_obs, reward+new_reward, done, info


if __name__ == "__main__":

    import time
    from env_tools.wrapper import MinigridFrameStacker

    env = SafeCrossing(size=7, bad_zone_action_proba=0, good_zone_action_proba=1,
                       proba_reset=0,
                       n_zone=99, feedback_when_wall_hit=0, reward_when_falling=0, n_more_actions=0, obstacle_type="none")
    env = MinigridFrameStacker(env, n_frameskip=3)

    save = set()

    for i in range(1000):

        env.reset()

        done = False
        step = 0
        sum_reward = 0

        while not done:

            env.render()

            a = env.action_space.sample()
            a = int(input(str(env.env.current_zone_num)+"\nHere : "))
            state, reward, done, info = env.step(action=a)
            
            print(state)

            sum_reward += reward

            step += 1

            env.render()


            #print(state, reward, done, info)

        if sum_reward > 0:
            save.add((step, sum_reward))



    save = list(save)
    save.sort(key=lambda x:x[1])
    print(save)
