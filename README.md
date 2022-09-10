# Adverserial Masking for Domain Generalization
> All the bash commands mentioned below shall be executed under the root directory of the repository.
## datasets
This folder contains the codes for preparing various Domain Generalization Datasets.
### Implemented Datasets
1. PACS
2. DomainNet
## mask
This folder contains the codes for the definition and the training of the advererial mask.
### Training
```bash
python mask/main.py
```
### Visualization
To obtain the visualization of a mask, run the following script:
``` bash
python mask/visualize.py
```
## classifier
This folder contains the codes for training and evaluating the classification performance of the mask augmented training.

### Training
```bash
python classifier/main.py
```
### Evaluation
```bash
python classifier/best_acc.py
```