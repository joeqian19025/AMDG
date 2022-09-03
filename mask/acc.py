from prepare import *
from mask import Mask, Classifier, MaskTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

if args.dataset == "pacs":
    from pacs import *

    args.num_classes = 7

    pacs_dataset = PACSWithVal(
        args.dataset_folder,
        args.test_envs,
        args.train_val_ratio,
    )

    trainset = torch.utils.data.ConcatDataset(pacs_dataset.trainsets)
    valset = torch.utils.data.ConcatDataset(pacs_dataset.valsets)

valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.batch_size,
    shuffle=True,
    sampler=None,
    num_workers=4,
)

trainloader = torch.utils.data.Dataloader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    sampler=None,
    num_workers=4,
)

for image, label in trainloader:
    image = image.to(args.device)
