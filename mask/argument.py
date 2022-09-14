import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--experiment_path", type=str, default="./mask/experiment_folder"
)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--mask_model", type=str, default="unet")
parser.add_argument("--mask_pretrained", type=str, default="True")
parser.add_argument(
    "--mask_path", type=str, default="./mask/unet/unet_carvana_scale0.5_epoch2.pth",
)
parser.add_argument("--bilinear", action="store_true", default=False)

parser.add_argument("--classifier", type=str, default="resnet18")
parser.add_argument("--classifier_pretrained", type=str, default="True")
parser.add_argument("--double_classifiers", action="store_true", default=False)

parser.add_argument("--dataset", type=str, default="pacs")
parser.add_argument("--dataset_folder", type=str, default="/scratch/local/ssd/tuan/data/")

parser.add_argument("--test_envs", nargs="*", type=int, default=[])
parser.add_argument("--train_val_ratio", type=float, default=0.9)

parser.add_argument("--mask_epochs", type=int, default=500)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=0.001)
