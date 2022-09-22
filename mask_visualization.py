import os

import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from torchvision import transforms

from mask.prepare import *
from mask.mask import Mask
from mask.utils import get_save_name

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
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(),]
)
for valset in dataset.valsets:
    valset.dataset = copy.copy(valset.dataset)
    valset.dataset.transform = transform
valset = torch.utils.data.ConcatDataset(dataset.valsets)

valloader = torch.utils.data.DataLoader(
    valset, batch_size=1, shuffle=True, sampler=None, num_workers=4,
)

save_name = get_save_name(args)

mask = Mask(args)
mask.load_state_dict(torch.load(save_name + "_mask.pt"))
mask = mask.to(args.device)
os.system(f"mkdir {save_name}_visualization")
mask.eval()

for i, (original_image, _) in enumerate(valloader):
    original_image = original_image.to(args.device)
    image = transforms.Normalize(
        mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225]
    )(original_image)
    image = image.to(args.device)
    mask_t = mask(image)
    original_masked_image = mask_t * original_image
    original_inverse_masked_image = (1 - mask_t) * original_image
    masked_image = mask_t * image
    inverse_masked_image = (1 - mask_t) * image
    grid = make_grid(
        [
            original_image[0],
            mask_t[0],
            original_masked_image[0],
            original_inverse_masked_image[0],
            image[0],
            mask_t[0],
            masked_image[0],
            inverse_masked_image[0],
        ],
        nrow=4,
        padding=25,
    )
    grid_image = ToPILImage()(grid)
    grid_image.save(f"{save_name}_visualization/{i:04}.jpg")
    print(f"Finished Proccessing {i:04}")
