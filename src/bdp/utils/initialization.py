import numpy as np


def xavier_init(m):
    s = np.sqrt(2. / (m.in_features + m.out_features))
    m.weight.data.normal_(0, s)
