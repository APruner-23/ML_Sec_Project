# ML_Sec_Project
# Universal Adversarial Examples on RobustBench CIFAR-10 Models

## Overview
This project explores the generation of **universal, untargeted adversarial examples** that simultaneously fool multiple robust image-classification models trained on CIFAR-10. We:

1. Load three pretrained RobustBench models and craft adversarial perturbations via an **ensemble FGSM**.
2. Verify that each perturbation fools the ensemble.
3. Collect the successful adversarials and evaluate their **transferability** on seven additional RobustBench models.
4. Visualize sample perturbations, report generation statistics, and plot transfer-success metrics.

## 📂 Repository Structure

├── Progetto_MLSec_Leandri_Pruner.ipynb  # Colab notebook with project code  
├── presentation.pdf              
└── README.md                   

## Requirements
- Python 3.7+  
- PyTorch  
- torchvision  
- numpy  
- matplotlib  
- scikit-learn  
- [RobustBench](https://github.com/RobustBench/robustbench)

## The FGSM Adversarial Attack

### What is FGSM?
The **Fast Gradient Sign Method** (FGSM) is a one-step adversarial attack that perturbs an input image `x` in the direction of the gradient of the loss with respect to the input. Concretely, for a model `f` and loss function `L(f(x), y)`, the adversarial example is:
`x_adv = x + ε * sign(∇_x L(f(x), y))`
where:
- `ε` is a small scalar controlling the perturbation size (under an L-infinity norm constraint),
- `sign(·)` is the element-wise sign function,
- `y` is the true class label.

Despite its simplicity, FGSM often generates highly effective adversarial examples.

### Our Ensemble-FGSM Implementation
In this project, we adapt FGSM to attack an **ensemble of three robust models** simultaneously. The key steps are:

1. **Forward pass through each model** to obtain logits `z1(x)`, `z2(x)`, `z3(x)`.
2. **Compute the ensemble logits** as the element-wise average:
z_ens(x) = `(z1(x) + z2(x) + z3(x)) / 3`
3. **Apply a custom margin loss** `L_margin(z_ens, y)` that measures how much the highest non-true-class logit exceeds the true-class logit.
4. **Backpropagate** this loss to compute the gradient `∇_x L_margin`.
5. **Take one FGSM step** to generate the adversarial example:
`x_adv = clip(x + ε * sign(∇_x L_margin), 0, 1)`
6. **Verify** that `x_adv` fools the **ensemble** (i.e., the ensemble’s predicted label changes).

## Usage
1. Open `adversarial_ensemble.ipynb` in Colab or Jupyter.
2. Run each cell in order. The notebook is divided into sections for:
 - Setup & imports
 - Loading the three “source” models
 - Data preparation (CIFAR-10 subset)
 - Defining the ensemble FGSM attack
 - Generating & visualizing adversarials
 - Loading seven “target” models
 - Evaluating transfer success & plotting results

## Authors
Milena Leandri - ID: 70/90/00519  
Alessandro Pruner - ID: 70/90/00502
