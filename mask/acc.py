from prepare import *
from mask import Mask, Classifier, MaskTrainer
from utils import *

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
    batch_size=args.batchsize,
    shuffle=True,
    sampler=None,
    num_workers=4,
)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batchsize,shuffle=True,sampler=None,num_workers=4,)

save_name = get_save_name(args)

mask = Mask(args)
classifier = Classifier(args)
mask_trainer = MaskTrainer(mask, classifier, args)
mask_trainer.load_state_dict(torch.load(save_name + ".pt"))
print("Model Loaded")
mask_trainer = mask_trainer.to(args.device)
print("Model Sent to Device")

print(f"train acc: {calc_acc(mask_trainer, trainloader, args.device)}")
print(f"train acc: {calc_acc(mask_trainer, valloader, args.device)}")
