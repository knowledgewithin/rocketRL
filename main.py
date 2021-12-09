from data_classes import Rocket, State
from rocket_sim import Simulator
from model import Model
from actorcritic import Actor, Critic
import numpy as np
import torch
from tqdm import tqdm

def train_online_step(sim, actor, optim_actor, critic, optim_critic, critic_loss, gamma=.9, do_print=False):
    if do_print: print()

    # actor takes an action
    state = torch.tensor(np.array(sim.get_state()))
    action, norm_dist = actor(state)
    # print('action: ', action)
    if do_print: print('action: ', action)
    sim.take_action(*action.detach().numpy())
    next_state = torch.tensor(np.array(sim.get_state()))
    # states.append(next_state)

    reward = sim.get_state_reward()
    if do_print: print('resulting state:', next_state)

    # critic predicts value of next state
    value_Sprime = critic(next_state)
    if do_print: print('V(t+1):', value_Sprime)
    target = reward + gamma * value_Sprime.squeeze()
    if do_print: print('target:', target)

    # TD error
    value_S = critic(state)
    td_error = target - value_S.squeeze()

    # might need to change where clipping occurs, loss is massive at the moment
    # actor_loss = -norm_dist.log_prob(action).sum() * td_error
    actor_loss = -torch.mean(norm_dist.log_prob(action) * td_error)
    # actor_loss = actor_loss(td_error)
    # if sim.s.py < 50:
    #     print("actor loss",actor_loss, "td error", td_error)
    # actor_loss = -norm_dist.log_prob(action).sum() * target if norm_dist else -1*target
    if do_print: print("td_error", td_error)
    if do_print: print("actor loss", actor_loss)
    optim_actor.zero_grad()
    actor_loss.backward(retain_graph=True)
    optim_actor.step()

    # critic_loss = torch.mean((td_error) ** 2)
    critic_loss = critic_loss(value_S, target)
    optim_critic.zero_grad()
    critic_loss.backward()
    optim_critic.step()
    return reward

epochs = 5000
rocket = Rocket()
model = Model()
t = 0
dt = .1

#hyperparams
gamma = 0.95

actor = Actor()
critic = Critic()
optim_actor = torch.optim.Adam(actor.parameters(), lr=0.0005)
optim_critic = torch.optim.Adam(critic.parameters(), lr=0.002)
critic_loss = torch.nn.HuberLoss()

pbar = tqdm(range(epochs), desc="starting sim")
last_rewards = []
last_epoch_rewards = []
for i in pbar:
    sim = Simulator(rocket=rocket, dt=dt)
    reward_total = 0
    # states = []
    # last_state = sim.get_state()
    # last_value = critic(torch.tensor(np.array(next_state)))
    do_print = i>=epochs-2
    if do_print:
        print()
        print(f"-------------EPOCH {i}----------------")
    epoch_reward = 0
    t = 0
    while not sim.final_state_reached():
        epoch_reward += train_online_step(sim, actor, optim_actor, critic, optim_critic, critic_loss, gamma=gamma, do_print=do_print)
        t += dt

    reward = sim.get_final_reward()
    last_rewards.append(reward)
    last_epoch_rewards.append(epoch_reward)
    if len(last_rewards) > 100:
        last_rewards.pop(0)
        last_epoch_rewards.pop(0)
    pbar.set_description(f"(t: {t}, r: {int(reward)}, avg: {int(np.average(last_rewards))}), (ep r: {int(epoch_reward)}, avg: {int(np.average(last_epoch_rewards))}) "
                         + f": {sim.print_state()}")

    if do_print:
        print("Final Reward:", reward)
    # assert False

    # reward = sim.get_final_reward()
    # #train critic for each state, using total reward
    # for state in states:
    #     critic_loss = torch.mean((target - value_S)**2)
    #     optim_critic.zero_grad()
    #     critic_loss.backward()
    #     optim_critic.step()