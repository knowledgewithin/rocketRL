from data_classes import Rocket, State
from rocket_sim import Simulator
from model import Model
from actorcritic import Actor, Critic
import numpy as np
import torch
from tqdm import tqdm

def train_online_step(sim, actor, optim_actor, critic, optim_critic, do_print=False):
    if do_print: print()

    # actor takes an action
    state = torch.tensor(np.array(sim.get_state()))
    action, norm_dist = actor(state)
    if do_print: print('action: ', action)
    sim.take_action(*action)
    next_state = sim.get_state()
    # states.append(next_state)

    reward = sim.get_state_reward()
    if do_print: print('resulting state:', next_state)

    # critic predicts value of next state
    value_Sprime = critic(torch.tensor(np.array(next_state)))
    if do_print: print('V(t+1):', value_Sprime)
    target = reward + gamma * value_Sprime.squeeze()
    if do_print: print('target:', target)

    # TD error
    value_S = critic(state)
    td_error = target - value_S.squeeze()

    # might need to change where clipping occurs, loss is massive at the moment
    actor_loss = -norm_dist.log_prob(action).sum() * td_error
    if do_print: print("td_error", td_error)
    if do_print: print("actor loss", actor_loss)
    optim_actor.zero_grad()
    actor_loss.backward(retain_graph=True)
    optim_actor.step()

    critic_loss = torch.mean((target - value_S) ** 2)
    optim_critic.zero_grad()
    critic_loss.backward()
    optim_critic.step()

epochs = 10000
rocket = Rocket()
model = Model()
t = 0

#hyperparams
gamma = 0.1

actor = Actor()
critic = Critic()
optim_actor = torch.optim.Adam(actor.parameters(), lr=0.00001)
optim_critic = torch.optim.Adam(critic.parameters(), lr=0.00056)

for i in tqdm(range(epochs)):
    sim = Simulator(rocket=rocket, dt=.5)
    reward_total = 0
    # states = []
    # last_state = sim.get_state()
    # last_value = critic(torch.tensor(np.array(next_state)))
    do_print = i>=epochs-2
    if do_print:
        print()
        print(f"-------------EPOCH {i}----------------")
    while not sim.final_state_reached():
        train_online_step(sim, actor, optim_actor, critic, optim_critic, do_print=do_print)

    if do_print:
        reward = sim.get_final_reward()
        print("Final Reward:", reward)

    # reward = sim.get_final_reward()
    # #train critic for each state, using total reward
    # for state in states:
    #     critic_loss = torch.mean((target - value_S)**2)
    #     optim_critic.zero_grad()
    #     critic_loss.backward()
    #     optim_critic.step()