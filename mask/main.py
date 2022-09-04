from datetime import datetime
import csv
import torch
import torch.nn as nn

from mask import Mask, Classifier, MaskTrainer
from prepare import *

# Prepare the Dataset and the Dataloaders
if args.dataset == "pacs":
    from pacs import *

    args.num_classes = 7

    pacs_dataset = PACSWithVal(
        args.dataset_folder, args.test_envs, args.train_val_ratio,
    )
    trainset = torch.utils.data.ConcatDataset(pacs_dataset.trainsets)

# Initialize the Mask Model
mask = Mask(args)

# Initialize the Classifier and Trainer in the Adverserial Training
classifier = Classifier(args)
mask_trainer = MaskTrainer(mask, classifier, args)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    mask_trainer = nn.DataParallel(mask_trainer)
mask_trainer.to(args.device)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batchsize,
    shuffle=True,
    sampler=None,
    num_workers=4,
)

save_name = "{}/{}_seed{}_env{}".format(
    args.experiment_path, args.dataset, args.seed, args.test_envs,
)

history = {
    "mask_loss": [],
    "classifier_loss": [],
}

# Start Training
for epoch in range(args.mask_epochs):
    mask_loss, classifier_loss = mask_trainer.train_epoch(trainloader)
    print(
        f"Epoch {epoch}: {datetime.now()}; Mask Loss: {mask_loss}; Classifier Loss: {classifier_loss}"
    )
    history["mask_loss"].append(mask_loss)
    history["classifier_loss"].append(classifier_loss)

    torch.save(mask_trainer.state_dict(), save_name + ".pt")

    with open(save_name + ".csv", "w") as fp:
        writer = csv.writer(fp)
        writer.writerow([key for key in history])
        for e in range(epoch):
            writer.writerow([history[key][epoch] for key in history])
