import torch

from data_classes import Rocket, State
import numpy as np

class Simulator:
    ''' A physics simulator for flying a rocket '''
    def __init__(self, rocket = None, state = None, dt = .05, g = -9.81):
        self.dt = dt
        self.rocket = rocket if rocket else Rocket()
        self.s = state if state else State()
        self.g = g

        # Penalty factors for the final state of each of (velocity, position, angle, and time)
        self.v_penalty = -1
        self.x_penalty = -1
        self.angle_penalty = -10    # its in radians...
        self.t_penalty = -5

    def get_state(self):
        return self.s

    def set_state(self, s):
        self.s = s

    def copy_state(self):
        return State(**self.s.__dict__)

    def get_state_ar(self):
        return torch.tensor(np.array(self.s))

    def print_state(self):
        return f"t={int(self.s.t)}, p=( {int(self.s.px)}, {int(self.s.py)}), v=({int(self.s.vx)}, {self.s.vy}), ang={int(self.s.orientation_angle)}, fuel={int(self.s.fuel_level)}"

    def get_state_arr(self):
        return [self.s.px, self.s.py, self.s.vx, self.s.vyself.s.v_angular, self.s.orientation_angle, self.s.fuel_level]

    def final_state_reached(self):
        return self.s.py <=0

    def take_action(self, thrust_proportion, gimble_angle=0, fin_angle=0):
        ''' Take an action, consisting of a thrust proportion, a gimble angle of the engine, and a fin angle for the fins on the head of the rocket. '''

        # Calculate center of mass and moment of inertia
        CoM, MoI = self.get_mass_vars()
        fin_forward, fin_ang_acc = self.get_fin_forces(fin_angle, CoM, MoI)

        # Check if we have enough fuel to burn the requested about, and calculate the amount of resulting thrust
        if self.s.fuel_level > 0:
            thrust_forward, thrust_ang_acc = self.get_thrust_forces(thrust_proportion, gimble_angle, CoM, MoI)
            fuel_burn = thrust_proportion * self.rocket.burn_rate * self.dt
            if fuel_burn > self.s.fuel_level:
                prop_burn = self.s.fuel_level / fuel_burn
                thrust_forward *= prop_burn
                thrust_ang_acc *= prop_burn
                self.s.fuel_level = 0
            else:
                self.s.fuel_level -= fuel_burn
        else:
            thrust_forward, thrust_ang_acc = 0, 0

        # Calculate thrust variables, and update position and velocity
        total_thrust_forward = thrust_forward + fin_forward
        total_ang_acc = thrust_ang_acc + fin_ang_acc
        self.update_p_and_v(total_thrust_forward, total_ang_acc)
        self.s.t += self.dt

    def get_mass_vars(self):
        ''' Calculate center of mass, and moment of inertia for the rocket, based on the current fuel level '''
        mass = self.s.fuel_level + self.rocket.dry_mass
        fuel_height = (self.s.fuel_level / self.rocket.start_fuel_mass) * self.rocket.fuel_height_prop * self.rocket.height
        fuel_CoM = fuel_height * .5 * self.s.fuel_level
        total_CoM = (fuel_CoM + self.rocket.rocket_CoM) / mass
        fuel_MoI = self.get_MoI(self.s.fuel_level, total_CoM, fuel_height)
        rocket_MoI = self.get_MoI(self.rocket.dry_mass, total_CoM, self.rocket.height)
        total_MoI = fuel_MoI + rocket_MoI

        return total_CoM, total_MoI

    def get_MoI(self, mass, c, h):
        ''' Moment of inertia formula '''
        if h==0: return 0
        return mass / (3 * h) * (h ** 3 + 3 * h * c ** 2 - 3 * h ** 2 * c)

    def get_fin_forces(self, fin_angle, CoM, MoI):
        ''' Get the forces on the rocket from the air resistance over the fins '''
        v_angle = np.arctan2(self.s.vx, 0) if self.s.vy == 0 else np.arctan(self.s.vx/self.s.vy)
        fin_v_angle = self.s.orientation_angle + fin_angle - v_angle

        f_fin = -1*(np.sin(fin_v_angle) * self.total_v()) ** 2 * self.rocket.fin_drag
        fin_forward = np.abs(np.sin(fin_angle)) * f_fin
        fin_lever_arm = self.rocket.height - self.rocket.fin_offset - CoM
        fin_torque = np.cos(fin_angle) * f_fin * fin_lever_arm
        fin_ang_acc = fin_torque / MoI

        return fin_forward, fin_ang_acc

    def total_v(self):
        return np.sqrt(self.s.vx ** 2 + self.s.vy ** 2)

    def total_p(self):
        return np.sqrt(self.s.px ** 2 + self.s.py ** 2)

    def get_thrust_forces(self, thrust_proportion, gimble_angle, CoM, MoI):
        ''' Calculate the forces from the thrust and gimpling actions '''
        thrust = self.rocket.full_thrust * thrust_proportion
        thrust_forward = np.cos(gimble_angle) * thrust
        thrust_torque = np.sin(gimble_angle) * thrust * CoM
        thrust_ang_acc = thrust_torque / MoI

        return thrust_forward, thrust_ang_acc

    def update_p_and_v(self, thrust_forward, ang_acc):
        ''' update position and velocity states '''
        mass = self.s.fuel_level + self.rocket.dry_mass
        total_thrust_x = np.sin(self.s.orientation_angle) * thrust_forward
        total_thrust_y = np.cos(self.s.orientation_angle) * thrust_forward

        d_v_angular = ang_acc * self.dt
        d_vx = total_thrust_x / mass * self.dt
        d_vy = (self.g + total_thrust_y / mass) * self.dt

        # Assuming that changes in vx, vy, and v_angular happen smoothly, the average vs over dt will be halfway between the old v and the new v
        self.s.orientation_angle += (self.s.v_angular + d_v_angular / 2) * self.dt
        self.s.px += (self.s.vx + d_vx / 2) * self.dt
        self.s.py += (self.s.vy + d_vy / 2) * self.dt

        self.s.v_angular += d_v_angular
        self.s.vx += d_vx
        self.s.vy += d_vy

    def get_final_reward(self):
        ''' Get the final reward for an episode '''
        return 1000 \
               + self.v_penalty*self.total_v() \
               + self.x_penalty*self.total_p() \
               + self.angle_penalty*np.abs(self.s.orientation_angle) \
               + self.t_penalty*self.s.t

    def get_state_reward(self):
        ''' get the reward for a single state.  If the state is a final state, use a multiple of the final reward function, otherwise use the state reward. '''
        if self.final_state_reached():
            final_r = self.get_final_reward()*20
            return final_r
        else:
            r =  1000 \
               + self.x_penalty*self.total_p() \
               + self.angle_penalty*np.abs(self.s.orientation_angle)
            return r