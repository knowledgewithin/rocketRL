from data_classes import Rocket, State
from rocket_sim import Simulator
from actorcritic import Actor, Critic
import numpy as np
import torch
from tqdm import tqdm

from sprite import run_vis


def train_actor(sim, actor, optim_actor, critic, gamma=.9):
    ''' Train the actor online for a single step.  The critic will be trained later, once the episode is completed. '''

    # actor takes an action
    state = torch.tensor(np.array(sim.get_state()))
    action, norm_dist = actor(state)
    sim.take_action(*action.detach().numpy())
    next_state = torch.tensor(np.array(sim.get_state()))

    # critic predicts value of next state
    value_Sprime = critic(next_state)
    target = gamma * value_Sprime.squeeze()

    # TD error
    value_S = critic(state)
    advantage = target - value_S.squeeze()

    # Losses
    actor_loss = -torch.mean(norm_dist.log_prob(action) * advantage)
    optim_actor.zero_grad()
    actor_loss.backward(retain_graph=True)
    optim_actor.step()

    return action, next_state, actor_loss


#hyperparams
epochs = 5000
rocket = Rocket()
t = 0
dt = .1
gamma = 0.95

# Set up Actor and Critic
actor = Actor()
critic = Critic()
optim_actor = torch.optim.Adam(actor.parameters(), lr=0.00005)
actor_loss = torch.nn.HuberLoss()
optim_critic = torch.optim.Adam(critic.parameters(), lr=0.0001)
critic_loss = torch.nn.HuberLoss()

# Training Loop
pbar = tqdm(range(epochs), desc="starting sim")
last_rewards = []
action_hist = []
for i in pbar:
    # Set up Simulator for each epoch
    sim = Simulator(rocket=rocket, dt=dt)
    initial_state = sim.copy_state()
    reward_total = 0
    prev_state = torch.tensor(np.array(sim.get_state()))
    states = [prev_state]
    actor_loss = []

    # for each epoch, take actions at each time step until final state is reached
    while not sim.final_state_reached():
        action, state, loss = train_actor(sim, actor, optim_actor, critic, gamma=gamma)
        states.append(state)
        action_hist.append(action_hist)
        actor_loss.append(float(loss))

    # Find final reward for this epoch
    epoch_reward = sim.get_final_reward()
    prev_reward = torch.tensor([epoch_reward], dtype=torch.float32)

    # Train critic in reverse, based on final reward, and the discounted reward at every prior step.
    states.reverse()
    for s in states:
        value_S = critic(s)
        critic_loss_i = critic_loss(value_S, prev_reward)
        optim_critic.zero_grad()
        critic_loss_i.backward()
        optim_critic.step()
        prev_reward *= gamma

    # Track average rewards for reporting
    last_rewards.append(epoch_reward)
    if len(last_rewards) > 100:
        last_rewards.pop(0)

    # Update progress bar with rewards and final state variables
    pbar.set_description(f"(t: {sim.s.t}, (ep r: {int(epoch_reward)}, avg: {int(np.average(last_rewards))}), max_actor_loss: {max(actor_loss)}"
                         + f": {sim.print_state()}")

    # Generate visualization for the most recent episode, if we are at a tracked episode.
    # This will wait for user input to show the visualization, then continue once the simulation is completed and closed.
    if i!=0 and i%500==0: run_vis(rocket, initial_state=initial_state, dt=dt, action_hist=action_hist)