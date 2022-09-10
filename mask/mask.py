import einops
import torch
import torch.nn as nn
import torchvision.models as models

from mask.unet import UNet
import torch.nn.functional as F
import mask.criterions.mask_loss as Loss

# The Mask Model
class Mask(nn.Module):
    def __init__(self, args):
        super(Mask, self).__init__()
        for name in args.__dict__:
            setattr(self, name, getattr(args, name))

        if self.mask_model == "unet":
            model = UNet(3, 2, args.bilinear)
            if self.mask_pretrained == "True":
                model.load_state_dict(torch.load(args.mask_path))
        self.model = model
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self, x):
        return einops.repeat(
            F.softmax(self.model(x))[:, 0, :, :], "n h w -> n c h w", c=3
        )


# The Classifier in the Adverserial Mask Training
class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        for name in args.__dict__:
            setattr(self, name, getattr(args, name))

        if self.classifier == "resnet18":
            if self.classifier_pretrained == "True":
                model = models.resnet18(pretrained=True)
            else:
                model = models.resnet18(pretrained=False)
        elif self.encoder == "resnet50":
            if self.encoder_pretrained == "True":
                model = models.resnet50(weights="IMAGENET1K_V2")
            else:
                model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        self.model = model
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self, x):
        return self.model(x)


# The Wrapper Class for the Training of the Mask
class MaskTrainer(nn.Module):
    def __init__(self, mask, classifier, args):
        super(MaskTrainer, self).__init__()
        for name in args.__dict__:
            setattr(self, name, getattr(args, name))

        self.mask = mask
        self.classifier = classifier

        self.mask_optimizer = torch.optim.SGD(
            self.mask.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        self.mask_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.mask_optimizer, self.mask_epochs
        )

        self.classifier_optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        self.classifier_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.classifier_optimizer, self.mask_epochs
        )

    def train_epoch(self, dataloader):
        total_mask_loss = 0
        total_classifier_loss = 0

        for data, label in dataloader:
            data = data.to(self.device)
            label = label.to(self.device)
            mask = self.mask(data)
            masked_data = mask * data
            inverse_masked_data = (1 - mask) * data

            mask_loss = Loss.mask_loss(
                label,
                self.classifier(masked_data),
                self.classifier(inverse_masked_data),
                mask,
            )
            total_mask_loss = total_mask_loss + mask_loss
            self.mask_optimizer.zero_grad()
            mask_loss.backward()
            self.mask_optimizer.step()

            classifier_loss = Loss.classifier_loss(
                label,
                self.classifier(masked_data.detach()),
                self.classifier(inverse_masked_data.detach()),
            )
            total_classifier_loss = total_classifier_loss + classifier_loss
            self.classifier_optimizer.zero_grad()
            classifier_loss.backward()
            self.classifier_optimizer.step()

        return total_mask_loss, total_classifier_loss
