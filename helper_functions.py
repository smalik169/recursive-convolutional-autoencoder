import numpy as np


def exponetial_loss_scaling(epoch):
    def exp_l_scaling(sen_len):
        log_len = int(np.log2(sen_len))
        log_len = max(log_len, 5)
        exponent = max(0, 10 - log_len - max(0, epoch - log_len + 4))
        #exponent = max(0, 10 - log_len - max(0, epoch - log_len + 1))
        return 2 ** exponent
    return exp_l_scaling
