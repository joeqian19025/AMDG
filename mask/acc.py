from mask.prepare import *
from mask.mask import Mask, Classifier, MaskTrainer
from mask.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

if args.dataset == "pacs":
    from datasets.pacs import *

    args.num_classes = 7

    dataset = PACSWithVal(
        args.dataset_folder, args.test_envs, args.train_val_ratio,
    )
elif args.dataset == "domainNet":
    from datasets.domainNet import *

    args.num_classes = 345
    dataset = DomainNetWithVal(
        args.dataset_folder, args.test_envs, args.train_val_ratio
    )

trainset = torch.utils.data.ConcatDataset(dataset.trainsets)
valset = torch.utils.data.ConcatDataset(dataset.valsets)
testset = torch.utils.data.ConcatDataset(dataset.testsets)

valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.batchsize,
    shuffle=True,
    sampler=None,
    num_workers=4,
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batchsize,
    shuffle=True,
    sampler=None,
    num_workers=4,
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batchsize,
    shuffle=True,
    sampler=None,
    num_workers=4,
)

save_name = get_save_name(args)

mask = Mask(args)
classifier = Classifier(args)
mask_trainer = MaskTrainer(mask, classifier, args)
mask_trainer.load_state_dict(torch.load(save_name + ".pt"))
print("Model Loaded")
mask_trainer = mask_trainer.to(args.device)
print("Model Sent to Device")

print(f"train acc: {calc_mask_acc(mask_trainer, trainloader, args.device)}")
print(f"val acc: {calc_mask_acc(mask_trainer, valloader, args.device)}")
print(f"test acc: {calc_test_acc(mask_trainer, testloader, args.device)}")
