from data_classes import Rocket, State
from rocket_sim import Simulator
from actorcritic import Actor, Critic
import numpy as np
import torch
from tqdm import tqdm

from sprite import run_vis


def train_online_step(sim, actor, optim_actor, critic, optim_critic, critic_loss, gamma=.9):
    ''' Train a single step for this online method '''

    # actor takes an action
    state = torch.tensor(np.array(sim.get_state()))
    action, norm_dist = actor(state)
    sim.take_action(*action.detach().numpy())
    next_state = torch.tensor(np.array(sim.get_state()))

    # get reward
    reward = sim.get_state_reward()

    # critic predicts value of next state
    value_Sprime = critic(next_state)
    target = reward + gamma * value_Sprime.squeeze()

    # TD error
    value_S = critic(state)
    td_error = target - value_S.squeeze()

    # losses
    actor_loss = -torch.mean(norm_dist.log_prob(action) * td_error)
    optim_actor.zero_grad()
    actor_loss.backward(retain_graph=True)
    optim_actor.step()

    critic_loss = critic_loss(value_S, target)
    optim_critic.zero_grad()
    critic_loss.backward()
    optim_critic.step()
    return action, reward



#hyperparams
epochs = 50000
rocket = Rocket()
t = 0
dt = .1
gamma = 0.95

# Set up Actor and Critic
actor = Actor()
critic = Critic()
optim_actor = torch.optim.Adam(actor.parameters(), lr=0.0005)
optim_critic = torch.optim.Adam(critic.parameters(), lr=0.002)
critic_loss = torch.nn.HuberLoss()

# Training Loop
pbar = tqdm(range(epochs), desc="starting sim")
last_rewards = []
last_epoch_rewards = []
for i in pbar:
    # Set up Simulator for each epoch
    sim = Simulator(rocket=rocket, dt=dt)
    initial_state = sim.copy_state()
    reward_total = 0
    epoch_reward = 0
    t = 0
    action_hist = []

    # for each epoch, take actions at each time step until final state is reached
    while not sim.final_state_reached():
        action, epoch_reward_i = train_online_step(sim, actor, optim_actor, critic, optim_critic, critic_loss, gamma=gamma)
        action_hist.append(action)
        epoch_reward += epoch_reward_i
        t += dt

    # Get final reward for tracking, and calculate recent performance averages
    reward = sim.get_final_reward()
    last_rewards.append(reward)
    last_epoch_rewards.append(epoch_reward)
    if len(last_rewards) > 100:
        last_rewards.pop(0)
        last_epoch_rewards.pop(0)

    # Update progress bar with rewards and final state variables
    pbar.set_description(f"(t: {t}, r: {int(reward)}, avg: {int(np.average(last_rewards))}), (ep r: {int(epoch_reward)}, avg: {int(np.average(last_epoch_rewards))}) "
                         + f": {sim.print_state()}")

    # Generate visualization for the most recent episode, if we are at a tracked episode.
    # This will wait for user input to show the visualization, then continue once the simulation is completed and closed.
    if i!=0 and i>48000 and i%100==0: run_vis(rocket, initial_state=initial_state, dt=dt, action_hist=action_hist)
