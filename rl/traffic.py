import gym
from gym.utils import seeding
#from gym.envs.classic_control.rendering import SimpleImageViewer
from rl.rendering import SimpleImageViewer
from gym import spaces

import numpy as np

DT = 0.0025
SWD = 300
SHD = 600


class TrafficEnv(gym.Env):
    def __init__(self, nlanes, ncars, images=True, sh=50):
        self.ncars = ncars
        self.nlanes = nlanes
        self.images = images

        self.l_lims = np.array((0.0, 0.0)) # f, b
        self.h_lims = np.array((1.0, 1.0)) # f, b
        self.vva = 0.005
        self.vlims = (0.01, 0.02)

        self.rsx = SWD/SHD
        self.sh = sh
        self.sw = int(self.sh * self.rsx)

        self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Box(self.l_lims, self.h_lims, dtype=float)
        ))

        if self.images:
            self.observation_space = spaces.Box(0.0, 1.0, shape=(self.sw, self.sh))
        else:
            obs_low = np.array((0.0, 0.0, -self.vva) +
                               (-self.rsx, -2.0, 0.0, -0.2)*(self.ncars-1))

            obs_high = np.array((self.rsx, 0.2, self.vva) +
                                (self.rsx, 2.0, self.rsx, 0.2)*(self.ncars-1))
            self.observation_space = spaces.Box(obs_low, obs_high)

        self.lanes = (np.arange(self.nlanes) + 0.5) * (self.sw/self.sh) / self.nlanes
        self.hlanes = (np.arange(self.nlanes+1) + 0.0) * ((self.sw-1)/self.sh) / self.nlanes

        self.viewer = None
        self.cars = []
        self.is_final = False
        self.lanev = None
        self.reward_func = None

        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        self.is_final = False
        car0y = 0.5
        self.lanev = np.random.rand(self.nlanes)*(self.vlims[1]-self.vlims[0]) + self.vlims[0]
        self.cars = [Car(self.lanes, int(self.nlanes/2), car0y, self.lanev[int(self.nlanes/2)])]
        for ii in range(1, self.ncars):
            self.cars.append(self.get_car())
        return self.road_img(car0y - 0.5, self.sw, self.sh)[0] if self.images else self.get_state(), dict()

    def get_car(self, ymin=0, ymax=1, lane=None):
        c_lane = np.random.randint(self.nlanes) if lane is None else lane
        c = Car(self.lanes, c_lane, np.random.rand() * (ymax - ymin) + ymin, self.lanev[c_lane])
        while self.car_overlaps(c):
            c_lane = np.random.randint(self.nlanes) if lane is None else lane
            c = Car(self.lanes, c_lane, np.random.rand() * (ymax - ymin) + ymin, self.lanev[c_lane])
        return c

    def car_overlaps(self, c, margin=1.2):
        for ii in range(len(self.cars)):
            if (self.cars[ii].lane == c.lane) and \
                    (np.abs(self.cars[ii].py - c.py) < (margin*(self.cars[ii].sy + c.sy))):
                return True
        return False

    def step(self, action):
        assert self.action_space[0].contains(action[0]), 'Action {} is invalid.'.format(action[0])
        assert self.action_space[1].contains(action[1]), 'Action {} is invalid.'.format(action[1])

        if not self.is_final:
            if action[0] != 1:
                self.cars[0].va = (action[0] - 1) * self.vva
            for it in range(200):
                car0y = self.cars[0].py
                for c in self.cars[1:]:
                    c.step(0.0, 1.0)
                self.cars[0].step(action[1][0]*0.0005, (1.0-action[1][1])*0.0001 + 0.99989)

            car0y = self.cars[0].py
            for ii in range(1,len(self.cars)):
                if (self.cars[ii].py - car0y) > 1:
                    self.cars[ii] = self.get_car(car0y-0.5, car0y-1.0)
                elif (self.cars[ii].py - car0y) < -1:
                    self.cars[ii] = self.get_car(car0y+0.5, car0y+1.0)

            rimg_i, cimgs_i = self.road_img(car0y - 0.5, self.sw, self.sh)
            self.is_final = (self.num_collisions(cimgs_i) > 0.0) or \
                            ((self.cars[0].px-self.cars[0].sx) <= 0.0) or \
                            ((self.cars[0].px+self.cars[0].sx) >= self.cars[0].rsx)

            reward = 0.0 if self.reward_func is None else self.reward_func(self.cars[0].px/self.rsx,
                                                                           self.cars[0].v,
                                                                           self.cars[0].va != 0.0,
                                                                           self.is_final)
        else:
            rimg_i, cimgs_i = self.road_img(self.cars[0].py - 0.5, self.sw, self.sh)
            self.is_final = True
            reward = 0.0
        return rimg_i if self.images else self.get_state(), reward, int(self.is_final), False, {}

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = SimpleImageViewer()
        rimg_i = self.road_img(self.cars[0].py - 0.5, SWD, SHD)[0]
        img = np.transpose(np.stack([((1.0-rimg_i)*255).astype(np.uint8)]*3, axis=2), (1, 0, 2))[::-1, :, :]
        self.viewer.imshow(img)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def car_imgs(self, y0, swi, shi):
        return np.transpose(np.array([c.render(y0, swi, shi) for c in self.cars]), (1,2,0))

    def road_img(self, y0, swi, shi):
        cimgs_i = self.car_imgs(y0, swi, shi)
        lane_line = 1.0*(np.sin( ((np.arange(shi)/shi) + y0)*20*2*np.pi ) > 0)
        road_img = np.sum(cimgs_i[:, :, 1:]*0.25, axis=2) + cimgs_i[:, :, 0]*1.0
        for l in self.hlanes:
            road_img[int(l*shi), :] = np.maximum(road_img[int(l*shi), :], lane_line*0.75)
        return np.minimum(road_img, 1.0), cimgs_i

    def num_collisions(self, cimgs_i):
        return np.sum((np.sum(cimgs_i, axis=2) > 1.0).flatten())

    def get_state(self):
        p0y = self.cars[0].py
        p0x = self.cars[0].px
        v0y = self.cars[0].v
        return np.concatenate((np.array([p0x, v0y, self.cars[0].va]),) +
                              tuple(np.array([c.px - p0x, c.py - p0y, c.px, c.v - v0y]) for c in self.cars[1:]))


class Car:
    def __init__(self, lanes, lane, py, v, va=0.0, rsx=0.5):
        self.rsx = rsx
        self.lanes = lanes

        self.sx = 0.5 * 0.5 * self.rsx / self.lanes.shape[0]
        self.sy = 1.75 * self.sx

        self.lane = lane
        self.px = self.lanes[lane]
        self.py = py
        self.v = v
        self.va = va

    def step(self, f=0.01, b=0.999):
        self.px += DT*self.va
        self.py += DT*self.v

        self.v += DT*f
        self.v *= b

        for ln, lx in enumerate(self.lanes):
            reached_lane = (((self.px - lx) < 0.0) and ((self.px - lx) > -0.005) and (self.va > 0)) or \
                           (((self.px - lx) > 0.0) and ((self.px - lx) <  0.005) and (self.va < 0))
            if reached_lane:
                self.va = 0
                self.px = lx
                self.lane = ln
                break

    def render(self, y0, sw, sh):
        frx = np.arange(sw)
        frx = (frx <= ((self.px + self.sx) * sh)) * \
              (frx >= ((self.px - self.sx) * sh))

        fry = np.arange(sh)
        fry = (fry <= ((self.py + self.sy - y0) * sh)) * \
              (fry >= ((self.py - self.sy - y0) * sh))

        return (frx[:, np.newaxis] * fry[np.newaxis, :])*1.0
    