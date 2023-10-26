# CSLP-AE: A Contrastive Split-Latent Permutation Autoencoder Framework for Zero-Shot Electroencephalography Signal Conversion

This read-me will go through the installation of required packages and dependencies, as well as the usage of the code.

## Installation

Required packages and specific versions used in the paper are the following:
- [Python](https://www.python.org/downloads/) 3.9.14
- [PyTorch](https://pytorch.org/get-started/locally/) 2.0.0
- [Numpy](https://numpy.org/install/) 1.23.5
- [Scikit-learn](https://scikit-learn.org/stable/install.html) 1.2.0
- [XGBoost](https://xgboost.readthedocs.io/en/stable/install.html) 1.7.5
- [Seaborn](https://seaborn.pydata.org/installing.html) 0.12.2
- [tqdm](https://github.com/tqdm/tqdm) 4.64.1
- [Weights \& Biases](https://docs.wandb.ai/quickstart) 0.15.0
- [Matplotlib](https://matplotlib.org/stable/users/release_notes) 3.7.1

However, the code should be able to run with the newest version of the packages.

First install Python using a package manager. Then install PyTorch using the instructions from the [official website](https://pytorch.org/get-started/locally/). Then install the rest of the packages using pip:

```bash
pip install numpy scikit-learn xgboost seaborn tqdm wandb matplotlib
```

Weights \& Biases (wandb) is the only package requiring an account. We provide versions of the code without wandb logging in the `no_wandb` folder.

## Usage

First thing is data preparation. Data must be downloaded and the folder structure kept intact. The data can be downloaded from [the ERP Core repository](https://doi.org/10.18115/D5JW4R).
In the `data_preparation` folder the create_dataset.py file will create a Pickle file containing all examples, subject labels and task labels for the dataset. The file can be run with the following command:

```bash
python create_dataset.py
```

If you are using Weighs \& Biases, you must first login to your account using the following command:

```bash
wandb login
```

The `train.py` file will automatically log the data to the Weights \& Biases dashboard. If you use the `no_wandb` folder, you must provide a ```--results_save_dir <path>``` argument to the `train_no_wandb.py` file in addition to the other available arguments.

Below we provide a list of arguments to provide the `train.py` file:
- ```--data_dir```: Directory where the data is located. Default: './'.
- ```--model_save_dir```: Directory where the model is saved. Default: './'.
- ```--batch_size```: Batch size to use during training. Default: 256.
- ```--epochs```: Number of epochs to train for. Default: 200.
- ```--lr```: Learning rate to use during training. Default: 0.0001.
- ```--channels```: Number of channels to use in the encoder and decoder. Default: 256.
- ```--num_layers```: Number of layers to use in the encoder and decoder. Default: 4.
- ```--latent_dim```: Dimension of the latent space. Default: 64.
- ```--data_line```: Dataset to use. Default: 'simple'.
- ```--final_div_factor```: Final division factor to use in the one-cycle learning rate scheduler. Default: 10.
- ```--initial_lr```: Initial learning rate to use in the one-cycle learning rate scheduler. Default: 0.0001.
- ```--max_lr```: Maximum learning rate to use in the one-cycle learning rate scheduler. Default: 0.0001.
- ```--pct_start```: At which point does the learning rate start to decrease. Default: 0.5.
- ```--random_seed```: Whether to use random seed. Default: 1.
- ```--use_tqdm```: Whether to use tqdm. Default: 1.
- ```--group```: Group to use for wandb. Default: ''.
- ```--override_seed```: Provide seed as override. Default: None.
- ```--extra_tags```: Extra tags to use for wandb. Default: ''.
- ```--full_eval```: Whether to perform classification in off-spaces. Default: 1.
- ```--eval_every```: How often to perform evaluation. Default: 70.
- ```--save_model```: Whether to save the model. Default: 1.
- ```--add_name```: Additional name to use for saving the model. Default: ''.
- ```--conversion_N```: Number of samples to use for conversion. Default: 2000.
- ```--extra_classifiers```: Whether to train addition KNN and ExtraTrees classifiers. Default: 1.
- ```--conversion_results```: Whether to measure the ERP conversion loss. Default: 1.

Some options are available, but not explored through the paper, these are the following:
- ```--recon_type```: Reconstruction loss type. Only 'mse' is considered in paper. Default: 'mse'.
- ```--content_cosine```: Whether to use cosine similarity for the content loss. Only. Default: 1.
- ```--sched_mode```: Scheduler mode. Default: 'max'.
- ```--sched_patience```: Scheduler patience. Default: 5.
- ```--sched_factor```: Scheduler factor. Default: 0.5.
- ```--old_sched```: Whether to use the old scheduler, old: cosine annealing, new: reduce on plateau. Cosine annealing was used in the paper. Default: 1.

Finally, the losses and weights are all controlled through the following arguments:
- ```--recon_enabled```: Whether to use the reconstruction loss. Default: 0.
- ```--recon_weight```: Weight of the reconstruction loss. Default: 1.0.
- ```--sub_contra_s_enabled```: Whether to use the contrastive loss in the subject latent space. Default: 0.
- ```--sub_contra_s_weight```: Weight of the contrastive loss in the subject latent space. Default: 1.0.
- ```--task_contra_t_enabled```: Whether to use the contrastive loss in the task latent space. Default: 0.
- ```--task_contra_t_weight```: Weight of the contrastive loss in the task latent space. Default: 1.0.
- ```--latent_permute_s_enabled```: Whether to use the same-subject latent-permutation loss. Default: 0.
- ```--latent_permute_s_weight```: Weight of the same-subject latent-permutation loss. Default: 1.0.
- ```--latent_permute_t_enabled```: Whether to use the same-task latent-permutation loss. Default: 0.
- ```--latent_permute_t_weight```: Weight of the same-task latent-permutation loss. Default: 1.0.
- ```--sub_cross_s_enabled```: Whether to use the supervised cross-entropy loss in the subject latent space. Default: 0.
- ```--sub_cross_s_weight```: Weight of the supervised cross-entropy loss in the subject latent space. Default: 1.0.
- ```--task_cross_t_enabled```: Whether to use the supervised cross-entropy loss in the task latent space. Default: 0.
- ```--task_cross_t_weight```: Weight of the supervised cross-entropy loss in the task latent space. Default: 1.0.

Weights are only applied if the corresponding loss is enabled. The quadruplet permutation loss can be enabled using the following arguments:
- ```--quadruplet_permute_enabled```: Whether to use the quadruplet permutation loss. Default: 0.
- ```--quadruplet_permute_F_enabled```: Whether to use the quadruplet permutation loss with expanded batch size. This is the one used in the supplementary materials Default: 0.
- ```--quadruplet_permute_weight```: Weight of the quadruplet permutation loss. Default: 1.0.

For the quadruplet permutation loss from the supplementary materials, both ```--quadruplet_permute_enabled``` and ```--quadruplet_permute_F_enabled``` should be set to 1.

# Reproduced training results
Training requires up to 14 GB of VRAM depending on the losses enabled.
To train a *CSLP-AE* model run the following command:
```bash
python train.py --sub_contra_s_enabled 1 --task_contra_t_enabled 1 --restored_permute_s_enabled 1 --restored_permute_t_enabled 1
```
To train a *SLP-AE* model run the following command:
```bash
python train.py --restored_permute_s_enabled 1 --restored_permute_t_enabled 1
```
To train a *C-AE* model run the following command:
```bash
python train.py --recon_enabled 1 --sub_contra_s_enabled 1 --task_contra_t_enabled 1
```
To train a *AE* model run the following command:
```bash
python train.py --recon_enabled 1
```
To train a *CE* model run the following command:
```bash
python train.py --sub_cross_s_enabled 1 --task_cross_t_enabled 1
```
To train a *CE(t)* model run the following command:
```bash
python train.py --task_cross_t_enabled 1
```

# Rerunning evaluation or conversion results

We provide files for reevaluating a model or for measuring the ERP conversion error.
Reevaluation only works with wandb.  `reevaluate.py` can be used to reevaluate a model. The following arguments are available:

- ```--ids```: Comma separated list of ids to reevaluate. Default: ''.
- ```--data_dir```: Data directory to use. Default: './'.
- ```--model_save_dir```: Model save directory to use. Default: './'.
- ```--save_dir```: Save directory to use. Default: './'.
- ```--wandb_directory```: Wandb directory to use. Default: 'converting-erps/'.

`reevaluate.py` will additionally provide t-SNE components for the latents on both training and test data. These can be used to reproduce the t-SNE plots from the paper.

`conversion_data.py` and `conversion_data_no_wandb.py` can be used to measure the ERP conversion error. The former uses wandb, the latter does not. The following arguments are common to both:
- ```--data_dir```: Data directory to use. Default: './'.
- ```--model_names```: Comma separated list of model names to use. Name must start with the wandb id followed by '-'. Default: ''.
- ```--conversion_N```: Number of samples to use for conversion. Default: 2000.

The following arguments are only available for `conversion_data.py`:
- ```--per_errors```: Whether to calculate the per-subject and per-paradigm conversion errors. Default: 1.

The following arguments are only available for `conversion_data_no_wandb.py`:
- ```--results_save_dir```: Save directory to use. Default: './'.
- ```--model_save_dir```: Model save directory to use. Default: './'.

  # Acknowledgments and Disclosure of Funding
  Funding in direct support of this work: Lundbeck Foundation grant R347-2020-2439.
