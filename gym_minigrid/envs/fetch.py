from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class FetchEnv(MiniGridEnv):
    """
    Environment in which the agent has to fetch a random object
    named using English text strings
    """

    def __init__(
        self,
        size=8,
        numObjs=3
    ):
        self.numObjs = numObjs

        super().__init__(
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)

        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

        types = ['key', 'ball']

        objs = []
        objs_type_color = []

        # For each object to be generated
        while len(objs) < self.numObjs:
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)

            if objType == 'key':
                obj = Key(objColor)
            elif objType == 'ball':
                obj = Ball(objColor)

            # to avoid having twice the same object.
            if not (objType, objColor) in objs_type_color:
                objs_type_color.append((objType, objColor))
                self.place_obj(obj)
                objs.append(obj)


        # Randomize the player start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        target = objs[self._rand_int(0, len(objs))]
        self.targetType = target.type
        self.targetColor = target.color

        descStr = '%s %s' % (self.targetColor, self.targetType)

        # Generate the mission string
        idx = self._rand_int(0, 5)
        if idx == 0:
            self.mission = 'get a %s' % descStr
        elif idx == 1:
            self.mission = 'go get a %s' % descStr
        elif idx == 2:
            self.mission = 'fetch a %s' % descStr
        elif idx == 3:
            self.mission = 'go fetch a %s' % descStr
        elif idx == 4:
            self.mission = 'you must fetch a %s' % descStr
        assert hasattr(self, 'mission')

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.carrying:
            if self.carrying.color == self.targetColor and \
               self.carrying.type == self.targetType:
                reward = self._reward()
                done = True
            else:
                reward = 0
                done = True

        return obs, reward, done, info


class FetchEnvFixed(FetchEnv):
    """
    Environment in which the agent has to fetch an single object among others
    """

    def __init__(
            self,
            size=6,
            numObjs=3,
            target_id=0
    ):
        self.numObjs = numObjs
        self.target_id = target_id

        assert self.target_id in (0, 1), "target must be 0 or 1 at the moment, target is '{}'".format(self.target_id)

        super().__init__(
            size=size,
            numObjs=numObjs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        types = ['key', 'ball']

        # Fixed object and color, those will be the 2 object to retrieve
        objs = [Key('blue'), Ball('red')]
        objs_type_color = [('key', 'blue'), ('ball', 'red')]

        # For each object to be generated
        while len(objs) < self.numObjs:
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)

            if objType == 'key':
                obj = Key(objColor)
            elif objType == 'ball':
                obj = Ball(objColor)

            # to avoid having twice the same object.
            if not (objType, objColor) in objs_type_color:
                objs_type_color.append((objType, objColor))
                objs.append(obj)

        # Generate object now
        for obj in objs:
            self.place_obj(obj)

        # Randomize the player start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        target = objs[self.target_id]
        self.targetType = target.type
        self.targetColor = target.color

        self.mission = 'nothing'

class FetchEnv5x5N2(FetchEnv):
    def __init__(self):
        super().__init__(size=5, numObjs=2)

class FetchEnv5x5N3(FetchEnv):
    def __init__(self):
        super().__init__(size=5, numObjs=3)

class FetchEnv6x6N2(FetchEnv):
    def __init__(self):
        super().__init__(size=6, numObjs=2)

class FetchEnvFixed5x5N3_2nd(FetchEnvFixed):
    def __init__(self):
        super().__init__(size=5, numObjs=3, target_id=0)

class FetchEnvFixed5x5N3_1st(FetchEnvFixed):
    def __init__(self):
        super().__init__(size=5, numObjs=3, target_id=1)


register(
    id='MiniGrid-Fetch-5x5-N2-v0',
    entry_point='gym_minigrid.envs:FetchEnv5x5N2'
)

register(
    id='MiniGrid-Fetch-5x5-N3-v0',
    entry_point='gym_minigrid.envs:FetchEnv5x5N3'
)

register(
    id='MiniGrid-Fetch-6x6-N2-v0',
    entry_point='gym_minigrid.envs:FetchEnv6x6N2'
)

register(
    id='MiniGrid-FixedFetch-5x5-N3-1st-v0',
    entry_point='gym_minigrid.envs:FetchEnvFixed5x5N3_1st'
)

register(
    id='MiniGrid-FixedFetch-5x5-N3-2nd-v0',
    entry_point='gym_minigrid.envs:FetchEnvFixed5x5N3_2nd'
)

register(
    id='MiniGrid-Fetch-8x8-N3-v0',
    entry_point='gym_minigrid.envs:FetchEnv'
)


