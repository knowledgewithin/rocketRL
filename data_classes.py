from dataclasses import dataclass

@dataclass
class Rocket:
    height: float = 50                  # height of the rocket in meters
    fin_offset: float = 3               # how far below the nose are the fins
    fuel_height_prop: float = .5        # What proportion of the rocket is the fuel tank
    fin_drag: float = 100              # the ratio of velicity to drag from the fins.  No idea what an appropriate value is.
    dry_mass: float = 100000            # wight of rocket without fuel in kg
    start_fuel_mass: float = 100000     # starting fuel mass in kg
    full_thrust: float = 7000000        # thrust in newtons if thrust is at 100%
    burn_rate: float = 20000          # burn rate in KG/s of 100% thrust

    def __post_init__(self):
        self.rocket_CoM = self.height * .5 * self.dry_mass

@dataclass
class State:
    px: float = 0                       # x position of rocket
    py: float = 1000                    # y position of rocket
    vx: float = 0                       # x velocity of rocket
    vy: float = 0                       # y velocity of rocket
    v_angular: float = 0                # angular velocity of rocket
    orientation_angle: float = 0        # orientation angle of rocket
    fuel_level: float = 0               # current fuel level (mass) in kg