from data_classes import Rocket, State
from rocket_sim import Simulator
from actorcritic import Actor, Critic
import numpy as np
import torch
from tqdm import tqdm
from sprite import run_vis

epochs = 5000
dt = .1
gamma = 0.95
rocket = Rocket()

actor = Actor()
critic = Critic()
optim_actor = torch.optim.Adam(actor.parameters(), lr=0.0001)
optim_critic = torch.optim.Adam(critic.parameters(), lr=0.0003)

pbar = tqdm(range(epochs), desc="starting sim")
last_rewards = []
for i in pbar:
    sim = Simulator(rocket=rocket, dt=dt)
    initial_state = sim.copy_state()
    log_probs = []
    values = []

    action_hist = []
    while not sim.final_state_reached():
        state = sim.get_state_ar()
        action, dist = actor(state)
        action_hist.append(action)
        value = critic(state)

        sim.take_action(*action.detach().numpy())
        next_state = sim.get_state_ar()

        log_prob = dist.log_prob(action).mean().unsqueeze(0)
        log_probs.append(log_prob)
        values.append(value)
        state = next_state

    reward = sim.get_final_reward()
    returns = torch.tensor([reward*gamma**i for i in range(len(values)-1,-1,-1)]).detach()
    values = torch.cat(values)
    advantage = returns - values
    log_probs = torch.cat(log_probs)

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    optim_actor.zero_grad()
    optim_critic.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    optim_actor.step()
    optim_critic.step()

    last_rewards.append(reward)
    if len(last_rewards) > 100: last_rewards.pop(0)

    pbar.set_description(f"(r: {int(reward)}, avg: {int(np.average(last_rewards))}), s: {sim.print_state()}")

    if i!=0 and i%500==0: run_vis(rocket, initial_state=initial_state, dt=dt, action_hist=action_hist)
