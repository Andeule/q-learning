import gym
import numpy as np

env = gym.make("MountainCar-v0")
# INITIALIZE VARIABLES

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

#show us we you are right now after every 2000 episodes
SHOW_EVERY = 200

# With this we cut our oberservation space into 20 junks
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
# Find out the size of each junk [0.09  0.007]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# q table shape (20, 20, 3)
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_descrete_state(current_state):
    discrete_state = (current_state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


discrete_state = get_descrete_state(env.reset())

print(q_table[discrete_state])

print(discrete_state)

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print("Episode: "+str(episode))
        render = True
    else:
        render = False

    discrete_state = get_descrete_state(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_descrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            # current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q) = learned value
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # Now we update the action from the q_table that we just took.
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"We made it on {episode}")
            env.render()
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

env.close()

# convert to a discrete sta te
