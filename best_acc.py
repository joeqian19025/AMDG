import pandas as pd
from utils import get_save_name
from argument import *

args = parser.parse_args()

df = pd.read_csv(f"{args.experiment_path}/" + get_save_name(args) + "_encoder.csv")
if len(df) > 100:
    df=df[:100]

best_idx = df.argmax(axis="val_acc")
print(f"best val accuracy achieved in epoch: {best_idx}")

best_val_acc = df.at[best_idx, "val_acc"]
print(f"val acc: {best_val_acc}")

test_acc = df.at[best_idx, "val_acc"]
print(f"test acc: {test_acc}")
