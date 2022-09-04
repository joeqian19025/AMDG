import torch
import torch.nn as nn
import torchvision.models as models

from criterions.encoder_loss import encoder_loss
from mask.mask import Mask
from utils import AverageMeter


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        for name in args.__dict__:
            setattr(self, name, getattr(args, name))

        if self.encoder == "resnet18":
            if self.encoder_pretrained == "True":
                model = models.resnet18(pretrained=True)
            else:
                model = models.resnet18(pretrained=False)
            self.encoder = torch.nn.Sequential(*(list(model.children())[:-1]))
            self.classifier = nn.Linear(model.fc.in_features, self.num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))


class ClassifierTrainer(nn.Module):
    def __init__(self, args):
        super(ClassifierTrainer, self).__init__()
        for name in args.__dict__:
            setattr(self, name, getattr(args, name))

        self.model = Classifier(args)
        self.mask = Mask(args)
        self.mask.load_state_dict(torch.load(args.mask_path))

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.encoder_epochs
        )

    def train_epoch(self, dataloader):
        total_loss = 0
        for image, label in dataloader:
            image = image.to(self.device)
            label = label.to(self.device)

            rep = self.model.encoder(image)
            mask_image = self.mask(image) * image
            mask_rep = self.model.encoder(mask_image.detach())

            pred = self.model.classifier(rep)
            mask_pred = self.model.classifier(mask_rep)

            loss = encoder_loss(rep, mask_rep, pred, mask_pred, label)
            total_loss = loss + total_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return total_loss

    def calc_acc(self, dataloader):
        acc_meter = AverageMeter()
        for image, label in dataloader:
            image = image.to(self.device)
            label = label.to(self.device)

            pred = self.model(image)

            acc = (torch.argmax(pred, 1) == label).float().mean()
            acc_meter.update(acc, image.shape[0])
        return acc_meter.average().item()
