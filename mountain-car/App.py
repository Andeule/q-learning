import gym

env = gym.make("MountainCar-v0")
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print("Number of actions: "+str(env.action_space.n))

# With this we cut our oberservation space into 20 junks
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
# Find out the size of each junk
discrete_os_win_size = (env.observation_space.high-env.observation_space.low) / DISCRETE_OS_SIZE
print("Discrete os size:"+str(DISCRETE_OS_SIZE))
print("Discrete os win size:" + str(discrete_os_win_size))


done = False

while not done:
    action = 2
    new_state, reward, done, _ = env.step(action)
    env.render()

env.close()
