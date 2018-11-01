import numpy as np

def get_cfg(max_ch):
    if max_ch == 8:
        cfg = {'enc': [8, 'D2', 8, 'D2', 8, 'D2', 8, 'D2', 8, 'D4'],
               'dec': ['U4', 8, 'U2', 8, 'U2', 8, 'U2', 8, 'U2', 8]}
    elif max_ch==16:
        cfg = {'enc': [16, 'D2', 16, 'D2', 16, 'D2', 16, 'D2', 16, 'D4'],
               'dec': ['U4', 16, 'U2', 16, 'U2', 16, 'U2', 16, 'U2', 16]}
    elif max_ch==32:
        cfg = {'enc': [16, 'D2', 32, 'D2', 32, 'D2', 32, 'D2', 32, 'D4'],
               'dec': ['U4', 32, 'U2', 32, 'U2', 32, 'U2', 32, 'U2', 16]}
    elif max_ch==64:
        cfg = {'enc': [16, 'D2', 32, 'D2', 64, 'D2', 64, 'D2', 64, 'D4'],
               'dec': ['U4', 64, 'U2', 64, 'U2', 64, 'U2', 32, 'U2', 16]}
    elif max_ch==128:
        cfg = {'enc': [16, 'D2', 32, 'D2', 64, 'D2', 128, 'D2', 128, 'D4'],
               'dec': ['U4', 128, 'U2', 128, 'U2', 64, 'U2', 32, 'U2', 16]}
    else:
        raise Exception('Invalid number of ch')

    return cfg

