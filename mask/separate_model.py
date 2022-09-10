from mask.prepare import *
from mask.mask import Mask, Classifier, MaskTrainer

args.num_classes = 7

save_name = "{}/{}_seed{}_env{}".format(
    args.experiment_path, args.dataset, args.seed, args.test_envs,
)

mask = Mask(args)
classifier = Classifier(args)
mask_trainer = MaskTrainer(mask, classifier, args)
mask_trainer.load_state_dict(torch.load(save_name + ".pt"))

torch.save(mask_trainer.mask.state_dict(), save_name + "_mask.pt")
torch.save(mask_trainer.classifier.state_dict(), save_name + "_classifier.pt")