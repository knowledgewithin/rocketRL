from data_classes import Rocket, State
from rocket_sim import Simulator
from model import Model
from actorcritic import Actor, Critic
import numpy as np
import torch

epochs = 1
rocket = Rocket()
model = Model()
sim = Simulator(rocket=rocket)
print("initial state:", sim.get_state())
t = 0
# while not sim.final_state_reached():
#     action = model.get_next_action(sim.get_state())
#     print(f"-----t={t}-----")
#     print("action",action)
#     sim.take_action(**action)
#     print("resulting",sim.get_state())
#     t+=1
#     print()

# reward = sim.get_reward()
# print("Final Reward:", reward)

#hyperparams
gamma = 0.1

actor = Actor()
critic = Critic()
optim_actor = torch.optim.Adam(actor.parameters(), lr=0.00001)
optim_critic = torch.optim.Adam(critic.parameters(), lr=0.00056)

for i in range(epochs):
    t = 0
    reward_total = 0
    while not sim.final_state_reached() or t < 10:
        # actor takes an action
        state = torch.tensor(np.array(sim.get_state()))
        action, norm_dist = actor(state)
        print('action: ', action)
        sim.take_action(*action)
        next_state = sim.get_state()
        reward = sim.get_reward()
        print('resulting state:', next_state)
        print('reward:', reward)

        reward_total += reward

        # critic predicts value of next state
        value_Sprime = critic(torch.tensor(np.array(next_state)))
        print('V(t+1):', value_Sprime)
        target = reward + gamma * value_Sprime.squeeze()
        print('target:', target)

        # TD error
        value_S = critic(state)
        td_error = target - value_S.squeeze()

        # might need to change where clipping occurs, loss is massive at the moment
        # TODO: how does the 3x1 get converted to a number (is it just sum?)
        actor_loss = -norm_dist.log_prob(action).sum() * td_error
        print(actor_loss)
        optim_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        optim_actor.step()

        critic_loss = torch.mean((target - value_S)**2)
        optim_critic.zero_grad()
        critic_loss.backward()
        optim_critic.step()


        t += 10