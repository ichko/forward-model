import gym
import time

# import keyboard


def play_mario():
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

    env = gym.make('SuperMarioBros-v3')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    env.reset()
    done = False
    step = -1

    while not done:
        step += 1
        time.sleep(1 / 100)
        env.render()

        # print(step)

        action = env.action_space.sample()
        # action = 0
        # if keyboard.is_pressed('a'):
        #     action = 4

        obs, reward, done, info = env.step(action)
        print(obs.shape)


if __name__ == '__main__':
    play_mario()
