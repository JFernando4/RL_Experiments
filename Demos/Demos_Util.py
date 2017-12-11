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
        average_return = np.average(rl_agent.return_per_episode[-episodes_per_iteration:])
        average_loss = np.average(rl_agent.fa.train_loss_history[-episodes_per_iteration:])

        print("### Results after", number_of_episodes, "episodes and", frame_count, "frames ###")
        print("Average Loss:", average_loss)
        print("Average Return:", average_return)

    if render:
        rl_agent.env.set_render(False)
    if agent_render:
        rl_agent.env.agent_render = False


def save_training_history(agent):
    " Agent's Variables and History "
    agent_dictionary = {"n": agent.n,
                        "gamma": agent.gamma,
                        "beta": agent.beta,
                        "sigma": agent.sigma,
                        "tpolicy": agent.tpolicy,
                        "bpolicy": agent.bpolicy,
                        "episode_number": agent.episode_number,
                        "return_per_episode": agent.return_per_episode,
                        "average_reward_per_timestep": agent.average_reward_per_timestep}

    " Environment's Variables and History "
    env_dictionary = {"frame_number": agent.env.frame_count,
                      "action_repeat": agent.env.action_repeat}

    " Model Variables "
    model_dictionary = {"name": agent.fa.model.model_name,
                        "dimensions": agent.fa.model.dimensions,
                        "dim_out": agent.fa.model.dim_out,
                        "loss_fun": agent.fa.model.loss_fun,
                        "gate_fun": agent.fa.model.gate_fun}

    " Function Approximator's Variables and History "
    fa_dictionary = {"num_actions": agent.fa.numActions,
                     "batch_size": agent.fa.batch_size,
                     "alpha": agent.fa.alpha,
                     "buffer_size": agent.fa.buffer_size,
                     "loss_history": agent.fa.train_loss_history,
                     "observation_dimensions": agent.fa.observation_dimensions}

    history = {"agent": agent_dictionary,
               "environment": env_dictionary,
               "model": model_dictionary,
               "fa": fa_dictionary}
    return history
