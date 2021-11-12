import numpy as np

class Model:
    def __init__(self):
        pass

    def get_next_action(self, state):
        return {
            'gimble_angle': np.random.uniform(-1*np.pi/4, np.pi/4),
            'thrust_proportion': np.random.uniform(.5, 1),
            'fin_angle': np.random.uniform(-1*np.pi/4, np.pi/4)
        }