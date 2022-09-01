import numpy as np
import time
import cv2
from Agent import Agent
from Runner import Runner
from Buffer import Buffer
import pickle

# Hyperparameters of the PPO algorithm
steps_per_epoch = int(5e2)  # 5e3
epochs = 300
gamma = 0.999
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 40
train_value_iterations = 40
lam = 0.97
target_kl = 0.01

windowname = "Flappy Bird New"
env_name = "Flappy_Bird_New"
state_dimensions = (4, 128, 128)
num_actions = 2

game_over_image = cv2.imread("game_over.bmp")
game_over_image = np.array(game_over_image)
agent = Agent(2, env_name, dl1=512, dl2=512, lr=0.001, ent_coef=0.01)
runner = Runner(agent, env_name)
# Initialize the buffer
buffer = Buffer(state_dimensions, steps_per_epoch, gamma=gamma)

load = False
# True if you want to render the environment

episode_return, runner.length = 0, 0
if load:
    state = runner.first_state()
    agent.critic(state)
    agent.actor(state)
    agent.load_models()

lib = 0  # Number of episodes stored in my library
# epochs from disc
if lib != 0:
    for epoch in range(lib):
        with open(f"buffer\epoch_{epoch}.pkl", 'rb') as f:
            mynewlist = pickle.load(f)

        state_buffer, action_buffer, advantage_buffer, return_buffer, logprobability_buffer, value_buffer = mynewlist
        sum_return, num_episodes, total_length = 0, 0, 0

        # Update the policy and implement early stopping using KL divergence
        for _ in range(train_policy_iterations):
            kl = agent.train_policy(state_buffer, action_buffer, logprobability_buffer, advantage_buffer)
            if kl > 1.5 * target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(train_value_iterations):
            agent.train_value_function(state_buffer, return_buffer, value_buffer)

        critic_loss, actor_loss, entropy_loss = agent.losses()
        mean_return = 0
        if agent.best_score < mean_return:
            agent.best_score = mean_return
            if epoch > 5:
                agent.save_models()

        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {mean_return}. Mean Length: {0}")
        print(
            f"Actor loss: {np.round(np.mean(actor_loss), 2)}, "
            f"Critic loss: {np.round(np.mean(critic_loss), 2)}, "
            f"Entropy loss: {np.round(np.mean(entropy_loss), 2)}\n"
            f"Entropy coef: {np.round(np.mean(agent.ent_coef), 4)}, "
            f"Probs = {agent.probs[:2]}"
        )

# Iterate over the number of epochs
for epoch in range(epochs):
    # if epoch != 0:
    #     agent.ent_coef *= 0.99
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    # state, episode_return, episode_length = runner.first_state(), 0, 0
    sum_return = 0
    total_length = 0
    num_episodes = 0
    runner.length = 0
    state = runner.first_state()
    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        t1 = time.time()
        runner.length += 1
        # Get the logits, action, and take one step in the environment
        logits, action = agent.choose_action(state)
        state_new, reward, done = runner.next_state(action[0].numpy())
        # print(f"{runner.length} - {reward}")
        episode_return += reward

        # Get the value and log-probability of the action
        value_t = agent.critic(state)
        logprobability_t = agent.logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(state, action, reward, value_t, logprobability_t)
        # print(buffer.state_buffer.shape)

        # Update the state
        state = state_new

        # Finish trajectory if reached to a terminal state
        terminal = done
        t2 = time.time()
        # print(f"FPS: {round(1 / (t2-t1) * 4, 2)}")
        if terminal or (t == steps_per_epoch - 1):
            # print("terminal")
            # print(f"Episode return = {episode_return}, length = {episode_length}")
            last_value = 0 if done else agent.critic(state)
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            total_length += runner.length
            num_episodes += 1
            state, episode_return, runner.length = runner.first_state(), 0, 0

    runner.open_main_menu()
    print(f"finished epoch {epoch + 0}")
    # Get values from the buffer
    (
        state_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
        value_buffer
    ) = buffer.get()
    #
    file = [
        state_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
        value_buffer
    ]
    with open(f"buffer\epoch_{epoch + lib}.pkl", "wb") as f:
        pickle.dump(file, f)
    # with open('buffer.pkl', 'rb') as f:
    #     mynewlist = pickle.load(f)
    #
    # state_buffer, action_buffer, advantage_buffer, return_buffer, logprobability_buffer, value_buffer = mynewlist
    # print(f"state_buffer shape is {state_buffer.shape}")
    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = agent.train_policy(state_buffer, action_buffer, logprobability_buffer, advantage_buffer)
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        agent.train_value_function(state_buffer, return_buffer, value_buffer)

    critic_loss, actor_loss, entropy_loss = agent.losses()
    mean_return = sum_return / num_episodes
    if agent.best_score < mean_return:
        agent.best_score = mean_return
        if epoch > 5:
            agent.save_models()

    runner.open_main_menu()

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + lib}. Mean Return: {mean_return}. Mean Length: {total_length / num_episodes}")
    print(
        f"Actor loss: {np.round(np.mean(actor_loss), 2)}, "
        f"Critic loss: {np.round(np.mean(critic_loss), 2)}, "
        f"Entropy loss: {np.round(np.mean(entropy_loss), 2)}\n"
        f"Entropy coef: {np.round(np.mean(agent.ent_coef), 4)}, "
        f"Probs = {agent.probs[:2]}"
    )
