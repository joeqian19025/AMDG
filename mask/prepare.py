import torch
import random
import numpy as np

from mask.argument import *

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
