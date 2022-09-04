class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0

    def update(self, val, n=1):
        self.count += n
        self.sum += val * n

    def average(self):
        return self.sum / self.count

    def __repr__(self):
        r = self.sum / self.count
        if r < 1e-3:
            return "{:.2e}".format(r)
        else:
            return "%.4f" % (r)


def get_save_name(args):
    return "{}_seed{}_env{}".format(
        args.dataset, args.seed, args.test_envs,
    )
