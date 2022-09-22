import os
import torch
from torchvision import transforms
from datasets.base import MultipleEnvironmentImageFolder
import copy


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNetWithVal(DomainNet):
    testsets = []
    trainsets = []
    valsets = []

    def __init__(self, root, test_envs, train_val_ratio):
        super().__init__(root, test_envs, {'data_augmentation': True})
        self.split_train_val_test(test_envs, train_val_ratio)

    def split_train_val_test(self, test_envs, train_val_ratio):
        self.testsets = []
        self.trainsets = []
        self.valsets = []
        transform = transforms.Compose([                                         
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.465, 0.406],
                std = [0.229, 0.224, 0.225]
                )
        ])
        for i, env_dataset in enumerate(self.datasets):
            if i in test_envs:
                self.testsets.append(env_dataset)
            else:
                N_train = int(train_val_ratio * len(env_dataset))
                trainset, valset = torch.utils.data.random_split(env_dataset, [N_train, len(env_dataset)-N_train])
                valset.dataset = copy.copy(valset.dataset)
                valset.dataset.transform = transform
                self.valsets.append(valset)
                self.trainsets.append(trainset)
