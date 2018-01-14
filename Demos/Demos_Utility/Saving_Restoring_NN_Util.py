import tensorflow as tf
import pickle

from Objects_Bases.Agent_Base import AgentBase


class NN_Agent_History:

    def __init__(self, experiment_path=None, restore=False):
        if experiment_path is None:
            pass
        else:
            self.history = None
            if restore:
                self.history = pickle.load(open(experiment_path+"_history.p", mode="rb"))

    @staticmethod
    def save_training_history(experiment_path, agent=AgentBase()):
        """" Agent's Variables and History """
        agent_dictionary = agent.get_agent_dictionary()

        " Environment's Variables and History "
        env_dictionary = agent.env.get_environment_dictionary()

        " Model Variables "
        model_dictionary = agent.fa.model.get_model_dictionary()

        " Function Approximator's Variables and History "
        fa_dictionary = agent.fa.get_fa_dictionary()

        history = {"agent": agent_dictionary,
                   "environment": env_dictionary,
                   "model": model_dictionary,
                   "fa": fa_dictionary}

        pickle.dump(history, open(experiment_path + "_history.p", mode="wb"))

    def load_nn_agent_dictionary(self):
        return self.history["agent"]

    def load_nn_agent_environment_dictionary(self):
        return self.history["environment"]

    def load_nn_agent_model_dictionary(self):
        return self.history["model"]

    def load_nn_agent_fa_dictionary(self):
        return self.history["fa"]


def save_graph(sourcepath, tf_sess):
    saver = tf.train.Saver()
    save_path = saver.save(tf_sess, sourcepath+".ckpt")
    print("Model Saved in file: %s" % save_path)


def restore_graph(sourcepath, tf_sess):
    saver = tf.train.Saver()
    saver.restore(tf_sess, sourcepath+".ckpt")
    print("Model restored.")
