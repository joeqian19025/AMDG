import torch


def get_save_name(args):
    return "{}/{}_seed{}_env{}".format(
        args.experiment_path, args.dataset, args.seed, args.test_envs,
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


def calc_acc(mask_trainer, dataloader, device):
    mask_acc_meter = AverageMeter()
    inverse_mask_acc_meter = AverageMeter()
    mask_trainer.eval()
    for image, label in dataloader:
        image, label = image.to(device), label.to(device)
        mask = mask_trainer.mask(image)
        mask_pred = mask_trainer.classifier(image * mask)
        inverse_mask_pred = mask_trainer.classifier((1 - mask) * image)
        mask_acc = (torch.argmax(mask_pred, 1) == label).float().mean()
        inverse_mask_acc = (
            (torch.argmax(inverse_mask_pred, 1) == label).float().mean()
        )
        mask_acc_meter.update(mask_acc, image.shape[0])
        inverse_mask_acc_meter.update(inverse_mask_acc, image.shape[0])
    return (
        mask_acc_meter.average().float(),
        inverse_mask_acc_meter.average().float(),
    )
