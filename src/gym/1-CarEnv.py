import gym
import numpy as np

# there are three actions in this environment 0=left, 1=nothing, 2=right
env = gym.make("MountainCar-v0")
env.reset()

print("env.observation_space.high:",env.observation_space.high)
print("env.observation_space.low:",env.observation_space.low)
print("env.action_space.n:",env.action_space.n)

# this makes sure the Q algorithm works on different env's 
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) 
discrete_os_win_size = (env.observation_space.high-env.observation_space.low) / DISCRETE_OS_SIZE

print("discrete_os_win_size:",discrete_os_win_size)

# build the Q lookup table with discretized observation space, (20,20) * 3, (x,y * #possibleActions)
# randomly assigns Q values in the beginning
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print("q_table.shape:",q_table.shape)

done = False

# step through the environment
while not done:
    action = 2
    # state is position and velocity
    newState, reward, done, infoDict = env.step(action)
    print("newState:",newState)
    print("reward:", reward)
    env.render()
env.close()