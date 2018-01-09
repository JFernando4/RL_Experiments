import tensorflow as tf
import pickle


class NN_Agent_History:

    def __init__(self, experiment_path=None, restore=False):
        if experiment_path is None:
            pass
        else:
            self.history = None
            if restore:
                self.history = pickle.load(open(experiment_path+"_history.p", mode="rb"))

    @staticmethod
    def save_training_history(agent, experiment_path, agent_extra_parameters=False, env_extra_parameters=False,
                              fa_extra_parameters=False, model_extra_parameters=False):
        """" Agent's Variables and History """
        agent_dictionary = {"n": agent.n,
                            "gamma": agent.gamma,
                            "beta": agent.beta,
                            "sigma": agent.sigma,
                            "tpolicy": agent.tpolicy,
                            "bpolicy": agent.bpolicy,
                            "episode_number": agent.episode_number,
                            "return_per_episode": agent.return_per_episode,
                            "average_reward_per_timestep": agent.average_reward_per_timestep}
        if agent_extra_parameters:
            agent_dictionary['extra_parameters_list'] = agent.extra_parameters_list

        " Environment's Variables and History "
        env_dictionary = {"frame_number": agent.env.frame_count,
                          "action_repeat": agent.env.action_repeat}
        if env_extra_parameters:
            env_dictionary['extra_parameters_list'] = agent.env.extra_parameters_list

        " Model Variables "
        model_dictionary = {"name": agent.fa.model.model_name,
                            "model_dimensions": agent.fa.model.model_dimensions,
                            "loss_fun": agent.fa.model.loss_fun,
                            "gate_fun": agent.fa.model.gate_fun}
        if model_extra_parameters:
            model_dictionary['extra_parameters_list'] = agent.fa.model.extra_parameters_list

        " Function Approximator's Variables and History "
        fa_dictionary = {"batch_size": agent.fa.batch_size,
                         "alpha": agent.fa.alpha,
                         "buffer_size": agent.fa.buffer_size,
                         "loss_history": agent.fa.train_loss_history}
        if fa_extra_parameters:
            fa_dictionary['extra_parameters_list'] = agent.fa.extra_parameters_list

        history = {"agent": agent_dictionary,
                   "environment": env_dictionary,
                   "model": model_dictionary,
                   "fa": fa_dictionary}

        pickle.dump(history, open(experiment_path + "_history.p", mode="wb"))

    def load_nn_agent_history(self, extra_parameters=False):
        agent_history = self.history['agent']
        n, gamma, beta, sigma, tpolicy, bpolicy, episode_number, return_per_episdoe, average_reward_per_timestep = \
            (agent_history['n'],
             agent_history['gamma'],
             agent_history['beta'],
             agent_history['sigma'],
             agent_history['tpolicy'],
             agent_history['bpolicy'],
             agent_history['episode_number'],
             agent_history['return_per_episode'],
             agent_history['average_reward_per_timestep'])
        if extra_parameters:
            extra_parameters_list = agent_history['extra_parameters_list']
            return n, gamma, beta, sigma, tpolicy, bpolicy, episode_number, return_per_episdoe, \
                   average_reward_per_timestep, extra_parameters_list
        return n, gamma, beta, sigma, tpolicy, bpolicy, episode_number, return_per_episdoe, average_reward_per_timestep

    def load_nn_agent_environment_history(self, extra_parameters=False):
        environment_history = self.history['environment']
        frame_number, action_repeat = (environment_history['frame_number'], environment_history['action_repeat'])
        if extra_parameters:
            extra_parameters_list = environment_history['extra_parameters_list']
            return frame_number, action_repeat, extra_parameters_list
        return frame_number, action_repeat

    def load_nn_agent_model_history(self, extra_parameters=False):
        model_history = self.history['model']
        name, model_dimensions, loss, gate = (model_history['name'],
                                              model_history['model_dimensions'],
                                              model_history['loss_fun'],
                                              model_history['gate_fun'])
        if extra_parameters:
            extra_parameters_list = model_history['extra_parameters_list']
            return name, model_dimensions, loss, gate, extra_parameters_list
        return name, model_dimensions, loss, gate

    def load_nn_agent_fa_history(self, extra_parameters=False):
        fa_history = self.history['fa']
        batch_size, alpha, buffer_size, loss_history = \
            (fa_history['batch_size'],
             fa_history['alpha'],
             fa_history['buffer_size'],
             fa_history['loss_history'])
        if extra_parameters:
            extra_parameters_list = fa_history['extra_parameters_list']
            return batch_size, alpha, buffer_size, loss_history, extra_parameters_list
        return batch_size, alpha, buffer_size, loss_history


def save_graph(sourcepath, tf_sess):
    saver = tf.train.Saver()
    save_path = saver.save(tf_sess, sourcepath+".ckpt")
    print("Model Saved in file: %s" % save_path)


def restore_graph(sourcepath, tf_sess):
    saver = tf.train.Saver()
    saver.restore(tf_sess, sourcepath+".ckpt")
    print("Model restored.")
