import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import copy

class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([                                         
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.465, 0.406],
                std = [0.229, 0.224, 0.225]
                )
        ])

        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale = (0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.465, 0.406],
                std = [0.229, 0.224, 0.225]
            )
        ])

        self.datasets = []
        for i, environment in enumerate(environments):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACSWithVal(PACS):
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
            transforms.Resize(224),
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
