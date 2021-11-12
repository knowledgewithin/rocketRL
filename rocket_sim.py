from data_classes import Rocket, State
import numpy as np

class Simulator:
    def __init__(self, rocket = None, state = None, dt = 1, g = -9.81):
        self.dt = dt
        self.rocket = rocket if rocket else Rocket()
        self.s = state if state else State(fuel_level=self.rocket.start_fuel_mass)
        self.g = g

        # rewards
        self.v_penalty = -1
        self.x_penalty = -1
        self.angle_penalty = -10    # its in radians...

    def get_state(self):
        return self.s

    def get_state_arr(self):
        return [self.s.px, self.s.py, self.s.vx, self.s.vyself.s.v_angular, self.s.orientation_angle, self.s.fuel_level]

    def final_state_reached(self):
        return self.s.py <=0

    def take_action(self, gimble_angle, thrust_proportion, fin_angle):
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

        total_thrust_forward = thrust_forward + fin_forward
        total_ang_acc = thrust_ang_acc + fin_ang_acc
        self.update_p_and_v(total_thrust_forward, total_ang_acc)

    def get_mass_vars(self):
        # Calculate mass, center of mass, and moment of inertia
        mass = self.s.fuel_level + self.rocket.dry_mass
        fuel_height = (self.s.fuel_level / self.rocket.start_fuel_mass) * self.rocket.fuel_height_prop * self.rocket.height
        fuel_CoM = fuel_height * .5 * self.s.fuel_level
        total_CoM = (fuel_CoM + self.rocket.rocket_CoM) / mass
        fuel_MoI = self.get_MoI(self.s.fuel_level, total_CoM, fuel_height)
        rocket_MoI = self.get_MoI(self.rocket.dry_mass, total_CoM, self.rocket.height)
        total_MoI = fuel_MoI + rocket_MoI

        return total_CoM, total_MoI

    def get_MoI(self, mass, c, h):
        if h==0: return 0
        return mass / (3 * h) * (h ** 3 + 3 * h * c ** 2 - 3 * h ** 2 * c)

    def get_fin_forces(self, fin_angle, CoM, MoI):
        # TODO: This will depend on angular velocity also
        v_angle = np.arctan2(self.s.vx, self.s.vy)
        fin_v_angle = self.s.orientation_angle + fin_angle - v_angle

        f_fin = -1*(np.cos(fin_v_angle) * self.total_v()) ** 2 * self.rocket.fin_drag
        fin_forward = np.abs(np.cos(fin_angle)) * f_fin
        fin_lever_arm = self.rocket.height - self.rocket.fin_offset - CoM
        fin_torque = np.sin(fin_angle) * f_fin * fin_lever_arm
        fin_ang_acc = fin_torque / MoI

        return fin_forward, fin_ang_acc

    def total_v(self):
        return np.sqrt(self.s.vx ** 2 + self.s.vy ** 2)

    def get_thrust_forces(self, thrust_proportion, gimble_angle, CoM, MoI):
        thrust = self.rocket.full_thrust * thrust_proportion
        thrust_forward = np.cos(gimble_angle) * thrust
        thrust_torque = np.sin(gimble_angle) * thrust * CoM
        thrust_ang_acc = thrust_torque / MoI

        return thrust_forward, thrust_ang_acc

    def update_p_and_v(self, thrust_forward, ang_acc):
        # TODO: handle end case, where the ground is hit.  Only some of the acceleration will happen before impact.
        # TODO: angular velocity will also affect this, as over dt the rotation will cause a curve in thrust
        mass = self.s.fuel_level + self.rocket.dry_mass
        total_thrust_x = np.cos(self.s.orientation_angle) * thrust_forward
        total_thrust_y = np.sin(self.s.orientation_angle) * thrust_forward

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

    def get_reward(self):
        return self.v_penalty*self.total_v() \
               + self.x_penalty*self.s.px \
               + self.angle_penalty*np.abs(self.s.orientation_angle)