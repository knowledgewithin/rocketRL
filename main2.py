from data_classes import Rocket, State
from rocket_sim import Simulator
from model import Model
from actorcritic import Actor, Critic
import numpy as np
import torch
from tqdm import tqdm

def train_actor(sim, actor, optim_actor, critic, gamma=.9, do_print=False):
    if do_print: print()

    # actor takes an action
    state = torch.tensor(np.array(sim.get_state()))
    action, norm_dist = actor(state)
    # print('action: ', action)
    if do_print: print('action: ', action)
    sim.take_action(*action.detach().numpy())
    next_state = torch.tensor(np.array(sim.get_state()))
    # states.append(next_state)

    if do_print: print('resulting state:', next_state)

    # critic predicts value of next state
    value_Sprime = critic(next_state)
    if do_print: print('V(t+1):', value_Sprime)
    target = gamma * value_Sprime.squeeze()
    if do_print: print('target:', target)

    # TD error
    value_S = critic(state)
    advantage = target - value_S.squeeze()

    # might need to change where clipping occurs, loss is massive at the moment
    # actor_loss = -norm_dist.log_prob(action).sum() * td_error
    actor_loss = -torch.mean(norm_dist.log_prob(action) * advantage)
    # actor_loss = actor_loss(td_error)
    # if sim.s.py < 50:
    #     print("actor loss",actor_loss, "td error", td_error)
    # actor_loss = -norm_dist.log_prob(action).sum() * target if norm_dist else -1*target
    if do_print: print("td_error", advantage)
    if do_print: print("actor loss", actor_loss)
    optim_actor.zero_grad()
    actor_loss.backward(retain_graph=True)
    optim_actor.step()

    return next_state, actor_loss

epochs = 5000
rocket = Rocket()
model = Model()
t = 0
dt = .1

#hyperparams
gamma = 0.95

actor = Actor()
critic = Critic()
optim_actor = torch.optim.Adam(actor.parameters(), lr=0.00005)
actor_loss = torch.nn.HuberLoss()
optim_critic = torch.optim.Adam(critic.parameters(), lr=0.0001)
critic_loss = torch.nn.HuberLoss()

pbar = tqdm(range(epochs), desc="starting sim")
last_rewards = []
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
    prev_state = torch.tensor(np.array(sim.get_state()))
    states = [prev_state]
    actor_loss = []
    while not sim.final_state_reached():
        state, loss = train_actor(sim, actor, optim_actor, critic, gamma=gamma, do_print=do_print)
        states.append(state)
        actor_loss.append(float(loss))


    epoch_reward = sim.get_final_reward()
    prev_reward = torch.tensor([epoch_reward], dtype=torch.float32)
    states.reverse()
    for s in states:
        # critic_loss = torch.mean((td_error) ** 2)
        value_S = critic(s)
        critic_loss_i = critic_loss(value_S, prev_reward)
        optim_critic.zero_grad()
        critic_loss_i.backward()
        optim_critic.step()
        prev_reward *= gamma

    last_rewards.append(epoch_reward)
    if len(last_rewards) > 100:
        last_rewards.pop(0)
    pbar.set_description(f"(t: {sim.s.t}, (ep r: {int(epoch_reward)}, avg: {int(np.average(last_rewards))}), max_actor_loss: {max(actor_loss)}"
                         + f": {sim.print_state()}")
    # assert False

    # reward = sim.get_final_reward()
    # #train critic for each state, using total reward
    # for state in states:
    #     critic_loss = torch.mean((target - value_S)**2)
    #     optim_critic.zero_grad()
    #     critic_loss.backward()
    #     optim_critic.step()