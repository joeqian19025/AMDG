import datetime
import csv
import torch

from model import ClassifierTrainer
from prepare import *
from utils import get_save_name

if args.dataset == "pacs":
    from pacs import *

    args.num_classes = 7

    pacs_dataset = PACSWithVal(
        args.dataset_folder, args.test_envs, args.train_val_ratio,
    )

    trainset = torch.utils.data.ConcatDataset(pacs_dataset.trainsets)
    valset = torch.utils.data.ConcatDataset(pacs_dataset.valsets)
    testset = torch.utils.data.ConcatDataset(pacs_dataset.testsets)

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

save_name = f"{args.experiment_path}/" + get_save_name(args) + "_encoder"
args.mask_path = f"{args.mask_path}/" + get_save_name(args) + "_mask.pt"

trainer = ClassifierTrainer(args)
trainer.to(args.device)

history = {
    "loss": [],
    "train_acc": [],
    "val_acc": [],
    "test_acc": [],
}

for epoch in range(args.mask_epochs):
    loss = trainer.train_epoch(trainloader)
    train_acc = trainer.calc_acc(trainloader)
    val_acc = trainer.calc_acc(valloader)
    test_acc = trainer.calc_acc(testloader)
    print(
        f"Epoch {epoch}: {datetime.now()}; Loss: {loss}; Train_acc: {train_acc}; Val_acc: {val_acc}; Test_acc:{test_acc}"
    )
    history["loss"].append(loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["test_acc"].append(test_acc)

    torch.save(trainer.model.state_dict(), save_name + ".pt")

    with open(save_name + ".csv", "w") as fp:
        writer = csv.writer(fp)
        writer.writerow([key for key in history])
        for e in range(epoch):
            writer.writerow([history[key][epoch] for key in history])
