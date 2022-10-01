import torch


def get_save_name(args):
    return "{}/{}_seed{}_env{}_beta{}_gamma{}_{}{}".format(
        args.experiment_path, args.dataset, args.seed, args.test_envs,
        args.beta, args.gamma,
        "doubleClassifiers" if args.double_classifiers else "",
        "_useIterations" if args.use_iterations else "",
    )


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


def calc_test_acc(mask_trainer, dataloader, device):
    test_acc_meter = AverageMeter()
    mask_trainer.eval()
    for image, label in dataloader:
        image, label = image.to(device), label.to(device)
        pred = mask_trainer.classifier(image)
        test_acc = (torch.argmax(pred, 1) == label).float().mean()
        test_acc_meter.update(test_acc, image.shape[0])
    return test_acc_meter.average().item()
