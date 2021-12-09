from dataclasses import dataclass
import numpy as np

@dataclass
class Rocket:
    ''' A rocket class, containing properties used in the physics simulation '''
    height: float = 50                  # height of the rocket in meters
    fin_offset: float = 3               # how far below the nose are the fins
    fuel_height_prop: float = .5        # What proportion of the rocket is the fuel tank
    fin_drag: float = 1              # the ratio of velicity to drag from the fins.  No idea what an appropriate value is.
    dry_mass: float = 100000            # wight of rocket without fuel in kg
    start_fuel_mass: float = 200000     # starting fuel mass in kg
    full_thrust: float = 1000000        # thrust in newtons if thrust is at 100%
    burn_rate: float = 1000             # burn rate in KG/s of 100% thrust

    def __post_init__(self):
        self.rocket_CoM = self.height * .5 * self.dry_mass

    def state(self):
        return

@dataclass
class State:
    ''' A state of a rocket, initialized with some randome selections '''
    px: float = np.random.randint(-100, 100)                      # x position of rocket
    py: float = 1000                                              # y position of rocket
    vx: float = np.random.randint(-10, 10)                        # x velocity of rocket
    vy: float = np.random.randint(-10, 0)                         # y velocity of rocket
    v_angular: float = 0                                          # angular velocity of rocket
    orientation_angle: float = 0                                  # orientation angle of rocket
    fuel_level: float = 100000                                    # current fuel level (mass) in kg
    t: float = 0

    def __array__(self):
        return np.array([self.px, self.py, self.vx, self.vy, self.v_angular, self.orientation_angle, self.fuel_level, self.t], dtype=np.float32)