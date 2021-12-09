import pygame
import random
import numpy as np

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (200, 200, 200)




#hyperparams
from data_classes import Rocket, State
from rocket_sim import Simulator


class Block(pygame.sprite.Sprite):
    def __init__(self, color, width, height):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()

class RocketOb(pygame.sprite.Sprite):
    def __init__(self, color, width=20, height=100):
        super().__init__()
        # self.image = pygame.Surface([width, height])
        self.orig_image = pygame.image.load("rocket.png").convert()
        self.orig_image = pygame.transform.scale(self.orig_image, (40, 40))
        self.image = self.orig_image
        # self.image.fill(color)
        self.rect = self.image.get_rect()

    def rot_center(self, angle, x, y):
        angle = float(angle)
        x = float(x)
        y = float(y)
        angle = angle * 180/np.pi
        rotated_image = pygame.transform.rotate(self.orig_image, angle)
        new_rect = rotated_image.get_rect(center=self.image.get_rect(center=(x, y)).center)

        self.image = rotated_image
        self.rect = new_rect


def run_vis(rocket, initial_state, dt = .05, g = -9.81, action_hist=None):
    sim = Simulator(rocket=rocket, state=initial_state, dt=dt, g=g)
    # Initialize Pygame
    scale = 5
    pygame.init()
    screen_width = 1500
    screen_height = 800
    ground_y = screen_height-(100/scale)
    screen = pygame.display.set_mode([screen_width, screen_height])

    all_sprites_list = pygame.sprite.Group()

    state = sim.get_state()

    rocket_w = 20
    rocket_h = 100
    rob = RocketOb(BLACK, rocket_w, rocket_h)
    rob.rect.x = state.px + screen_width/2
    rob.rect.y = screen_height - state.py/2
    all_sprites_list.add(rob)

    ground = Block(GRAY, screen_width, ground_y)
    ground.rect.x = 0
    ground.rect.y = ground_y
    all_sprites_list.add(ground)

    # Loop until the user clicks the close button.
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    # -------- Main Program Loop -----------
    finished = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if not finished:
            br = True
            while br:
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        br=False
            for thrust, gimble, fins in action_hist:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                # thrust = random.random()
                # # thrust = 1
                # # thrust = 0
                # gimble = random.random()*np.pi/18-np.pi/36
                # # gimble = np.pi/360
                # gimble = 0
                # fins = random.random()*np.pi/2-np.pi/4
                # # fins = np.pi/4
                # print(thrust)

                # print(thrust, gimble, fins)
                sim.take_action(thrust, gimble, fins)

                x = state.px/scale + screen_width/2 - rocket_w/2
                y = ground_y - (rocket_h +state.py)/scale
                # print(state.orientation_angle, x, y)
                rob.rot_center(-1*state.orientation_angle, x, y)
                # print(state.px,state.py, state.orientation_angle)

                # Clear the screen
                screen.fill(WHITE)
                all_sprites_list.draw(screen)
                pygame.display.flip()

                # Limit to 60 frames per second
                # clock.tick(60)
            finished = True

    pygame.quit()


if __name__ == "__main__":
    dt=.2
    rocket = Rocket()
    sim = Simulator(rocket, dt=dt)
    initial_state = sim.copy_state()
    action_hist = []
    while not sim.final_state_reached():
        thrust = random.random()
        gimble = random.random()*np.pi/18-np.pi/36
        fins = random.random()*np.pi/2-np.pi/4
        action = (thrust, gimble, fins)
        sim.take_action(*action)
        action_hist.append(action)
        state = sim.get_state()

    print("final state reached, running vis")
    print(len(action_hist))
    run_vis(rocket, initial_state=initial_state, dt=dt, action_hist=action_hist)

