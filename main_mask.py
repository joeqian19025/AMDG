from datetime import datetime
import csv
from mask.utils import get_save_name
import torch
import torch.nn as nn

from mask.mask import MaskTrainer
from mask.prepare import *

# Prepare the Dataset and the Dataloaders
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
print(f"Dataset Loaded, train: {len(trainset)}, val: {len(valset)}")

# Initialize the Classifier and Trainer in the Adverserial Training
mask_trainer = MaskTrainer(args)
mask_trainer.to(args.device)
print("Model Initialized")

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batchsize,
    shuffle=True,
    sampler=None,
    num_workers=4,
)

valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.batchsize,
    shuffle=True,
    sampler=None,
    num_workers=4,
)

save_name = get_save_name(args)

history = {
    "mask_loss": [],
    "classifier_loss": [],
    "mask_acc": [],
    "unmask_acc": []
}

print("Start Training")
# Start Training
if args.use_iterations:
    iterations = 0
    while True:
        for data, label in trainloader:
            iterations += 1
            if iterations>args.mask_iterations:
                exit(0)
            mask_loss, classifier_loss = mask_trainer.train_iteration(data, label)
            print(f"Iteration {iterations}: Mask Loss: {mask_loss}; Classifier Loss: {classifier_loss}")
            history["mask_loss"].append(mask_loss)
            history["classifier_loss"].append(classifier_loss)
            torch.save(mask_trainer.mask.state_dict(), save_name + "_mask.pt")
            torch.save(
                mask_trainer.mask_classifier.state_dict(),
                save_name + "_mask_classifier.pt",
            )
            if args.double_classifiers:
                torch.save(
                    mask_trainer.unmask_classifier.state_dict(),
                    save_name + "_unmask_classifier.pt",
                )

            with open(save_name + ".csv", "w") as fp:
                writer = csv.writer(fp)
                writer.writerow([key for key in history])
                for e in range(iterations):
                    writer.writerow([history[key][e] for key in history])
else:
    for epoch in range(args.mask_epochs):
        mask_loss, classifier_loss = mask_trainer.train_epoch(trainloader)
        mask_acc, unmask_acc = mask_trainer.calc_mask_acc(valloader)
        print(
            f"Epoch {epoch}: {datetime.now()}; Mask Loss: {mask_loss}; Classifier Loss: {classifier_loss}; Mask Acc: {mask_acc}; Unmask Acc: {unmask_acc}"
        )
        history["mask_loss"].append(mask_loss)
        history["classifier_loss"].append(classifier_loss)
        history["mask_acc"].append(mask_acc)
        history["unmask_acc"].append(unmask_acc)

        torch.save(mask_trainer.mask.state_dict(), save_name + "_mask.pt")
        torch.save(
            mask_trainer.mask_classifier.state_dict(),
            save_name + "_mask_classifier.pt",
        )
        if args.double_classifiers:
            torch.save(
                mask_trainer.unmask_classifier.state_dict(),
                save_name + "_unmask_classifier.pt",
            )

        with open(save_name + ".csv", "w") as fp:
            writer = csv.writer(fp)
            writer.writerow([key for key in history])
            for e in range(epoch):
                writer.writerow([history[key][e] for key in history])
