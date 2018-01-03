import numpy as np


def training_loop(rl_agent, iterations=1, episodes_per_iteration=100, render=False, agent_render=False):
    if render:
        rl_agent.env.set_render(True)
    if agent_render:
        rl_agent.env.agent_render = True

    for i in range(iterations):
        rl_agent.train(episodes_per_iteration)
        number_of_episodes = rl_agent.episode_number
        frame_count = rl_agent.env.frame_count
        average_return = np.average(rl_agent.return_per_episode[:]) #[-episodes_per_iteration:])

        print("### Results after", number_of_episodes, "episodes and", frame_count, "frames ###")
        for key in rl_agent.fa.train_loss_history.keys():
            average_loss = np.average(rl_agent.fa.train_loss_history[key])  # [-episodes_per_iteration:])
            print("Average Loss from", key+":", average_loss)
        print("Average Return:", average_return)
        print("Return of Last Episode:", rl_agent.return_per_episode[-1])

    if render:
        rl_agent.env.set_render(False)
    if agent_render:
        rl_agent.env.agent_render = False
