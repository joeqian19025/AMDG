import os

import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from torchvision import transforms

from prepare import *
from mask import Mask, Classifier, MaskTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

if args.dataset == "pacs":
    from pacs import *

    args.num_classes = 7

    pacs_dataset = PACSWithVal(
        args.dataset_folder, args.test_envs, args.train_val_ratio,
    )
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(),]
    )
    for valset in pacs_dataset.valsets:
        valset.dataset = copy.copy(valset.dataset)
        valset.dataset.transform = transform
    valset = torch.utils.data.ConcatDataset(pacs_dataset.valsets)

valloader = torch.utils.data.DataLoader(
    valset, batch_size=1, shuffle=True, sampler=None, num_workers=4,
)

save_name = "{}/{}_seed{}_env{}".format(
    args.experiment_path, args.dataset, args.seed, args.test_envs,
)

mask = Mask(args)
classifier = Classifier(args)
mask_trainer = MaskTrainer(mask, classifier, args)
mask_trainer.load_state_dict(torch.load(save_name + ".pt"))
print("Model Loaded")
mask_trainer = mask_trainer.to(args.device)
print("Model Sent to Device")
os.system(f"mkdir {save_name}_visualization")
mask_trainer.eval()

for i, (original_image, _) in enumerate(valloader):
    original_image = original_image.to(args.device)
    image = transforms.Normalize(
        mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225]
    )(original_image)
    image = image.to(args.device)
    mask = mask_trainer.mask(image)
    original_masked_image = mask * original_image
    original_inverse_masked_image = (1 - mask) * original_image
    masked_image = mask * image
    inverse_masked_image = (1 - mask) * image
    grid = make_grid(
        [
            original_image[0],
            mask[0],
            original_masked_image[0],
            original_inverse_masked_image[0],
            image[0],
            mask[0],
            masked_image[0],
            inverse_masked_image[0],
        ],
        nrow=4,
        padding=25,
    )
    grid_image = ToPILImage()(grid)
    grid_image.save(f"{save_name}_visualization/{i:4}.jpg")
    print(f"Finished Proccessing {i:4}")
