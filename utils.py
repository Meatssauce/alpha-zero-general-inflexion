class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

    def __getstate__(self):
        # Convert the dictionary to a list of tuples (key, value) for serialization
        state = [(k, v) for k, v in self.items()]
        return state

    def __setstate__(self, state):
        # Deserialize the list of tuples (key, value) and update the dictionary
        self.update({k: v for k, v in state})

    def __getstate__(self):
        # Convert the dictionary to a list of tuples (key, value) for serialization
        state = [(k, v) for k, v in self.items()]
        return state

    def __setstate__(self, state):
        # Deserialize the list of tuples (key, value) and update the dictionary
        self.update({k: v for k, v in state})
