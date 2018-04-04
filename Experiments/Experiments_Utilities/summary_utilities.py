import numpy as np
import scipy.stats

""" Summary Functions """
# Confidence Interval
def compute_confidence_interval(sample_mean, sample_std, proportion, degrees_of_freedom):
    if degrees_of_freedom < 2:
        return None, None
    sample_size = degrees_of_freedom +1
    t_dist = scipy.stats.t(df=degrees_of_freedom)
    upper_tail = 1 - (1-proportion/2)
    lower_tail = 1 - upper_tail
    upper_factor = t_dist.ppf(upper_tail)
    lower_factor = t_dist.ppf(lower_tail)

    sqrt_inverse_sample_size = np.sqrt(1 / sample_size)
    upper_bound = sample_mean + (sample_std * upper_factor * sqrt_inverse_sample_size)
    lower_bound = sample_mean + (sample_std * lower_factor * sqrt_inverse_sample_size)

    return upper_bound, lower_bound
