from Import import *

def custom_loss_function(y_target,y_actual):
    return K.mean(K.square(y_target-y_actual))

def get_padding_sequence(sequence, t):
    size = sequence.size
    seq = sequence[:t]
    seq = np.append(seq, np.zeros(size-t))
    return seq