from data_classes import Rocket, State
from rocket_sim import Simulator
from model import Model

rocket = Rocket()
model = Model()
sim = Simulator(rocket=rocket)
print("initial state:", sim.get_state())

t = 0
while not sim.final_state_reached():
    action = model.get_next_action(sim.get_state())
    print(f"-----t={t}-----")
    print("action",action)
    sim.take_action(**action)
    print("resulting",sim.get_state())
    t+=1
    print()

reward = sim.get_reward()
print("Final Reward:", reward)