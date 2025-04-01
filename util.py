
import torch
import numpy as np


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def parse_years_from_index(index):
    parts = index.split('_')
    if len(parts)==3:
        return int(parts[1]), int(parts[2])
    else:
        return "Error", "Error" 

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import sys
class DualWriter:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
def create_windows(input_x, target, input_window=100, forecast_window=30, step=30):
    num_steps = input_x.shape[0]
    inputs = []
    targets = []
    start_list = []
    end_list = []
    for start in range(0, num_steps - input_window - forecast_window + 1, step):
        end = start + input_window
        input_window_data = input_x[start:end, :, :]
        target_window_data = target[end:end + forecast_window, :, :]
        inputs.append(input_window_data)
        targets.append(target_window_data)
        start_list.append(start)
        end_list.append(end + forecast_window)

    # Stack all windows to create a new dimension for windows
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    return inputs, targets, start_list, end_list