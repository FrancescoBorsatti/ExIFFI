# `ExIFFI_original`

This repository is a fork of
[`ExIFFI`](https://github.com/alessioarcudi/ExIFFI), the Anomaly Detection
interpretation method employed in the ["Towards Transparent and Efficient
Anomaly Detection in Industrial Processes through
ExIFFI"](https://arxiv.org/abs/2405.01158) paper.

In particular this experiment contains all the python script necessary to run
and execute the experiments, whose results are reported in the paper.

>[!info]
> This repository is used as a submodule of [the official paper's
> repository](https://github.com/FrancescoBorsatti/ExIFFI_Industrial_Test.git).

## Launch experiments

Experiments can be launched through some `bash` scripts, all contained in
`ExIFFI_original/experiments`.

### Global Importances Experiments

This experiment fits an `AD` model to the input dataset multiple times and
computes the Global Feature Importance score for each run. Finally the Score
Plot is produced to represent the average importance scores over the different
runs.

#### Example

There are several command line arguments that can be passed to the script but
the most important ones are:
- `dataset_name` → name of the benchmark dataset to use → e.g. `TEP_ACME,
piade_s2`
- `model_name` → name of the `AD` model to train → e.g. `EIF,EIF+,IF`
- `interpretation` → name of the interpretation algorithm to use → e.g.
`EXIFFI, EXIFFI+, DIFFI, ACME, KernelSHAP`

```bash
./launch_exp_GFI TEP_ACME EIF+ EXIFFI+
```

### Local Importances Experiments

This experiment works similarly to `launch_exp_GFI` but focuses on the Local
Feature Importance scores. Also in this case multiple runs with different seeds
are used and the average `LFI` scores for the test samples over the runs are
used to produce the `LFI` Score Plot.

#### Example

The command line arguments are the same used in `launch_exp_GFI`.

```bash
./launch_local_imp_exp TEP_ACME EIF+ EXIFFI+
```

### Local Scoremaps Experiments

This experiment produces the Local Scoremaps for a pair of features of a
dataset. The feature pairs used in the experiments reported in the paper is
contained in the file `ExIFFI_original/experiment/scoremaps_feats.json` and
it's the source used by the script to know which features to represent in the
scoremaps.

#### Example

The command line arguments are the same used in `launch_exp_GFI`.

```bash
./launch_scoremaps_exp TEP_ACME EIF+ EXIFFI+
```

### Feature Selection Experiments

This script performs the feature selection experiment and produces the plot
containing the results.

>[!warning]
> In order to successfully execute this script it is necessary to have saved the `GFI`
> matrices (which are needed to get the feature rankings). These matrices are automatically
> saved by the `launch_exp_GFI` script so make sure to run that first.

#### Example

The command line arguments are the same used in `launch_exp_GFI`.

```bash
./launch_fs_exp TEP_ACME EIF+ EXIFFI+
```

>[!warning]
> Note that this experiment cannot be executed on the `PIADE` dataset
> since there are no labels to compute the average precision metric.

### Ablation Studies

In this section we list the different scripts needed to produce the results of
the different kind of ablation studies.

>[!warning]
> The ablation studies were executed solely on the `EIF+_EXIFFI+` model and interpretation
> combination since their idea is to study the characteristics of the newly introduced method
> and not to perform comparisons with other models.

#### Ablation Trees

In this ablation study we study the trend of three variables in relation to the
number of trees used to fit the forest in `IF, EIF` and `EIF+` algorithms.
These variables are the average precision, the training and prediction times.
The script performs the experiment and produces one plot for each one of the
tracked variables.

##### Example

Since the focus is on `EIF+_EXIFFI+` the only command line argument for this
script is the dataset name:

```bash
./launch_ablation_tree TEP_ACME
```

>[!warning]
> Note that this experiment cannot be executed on the `PIADE` dataset
> since there are no labels to compute the average precision metric.

#### Ablation Contamination Prediction

In this experiment we track the ROC AUC metric as the contamination level
changes. As for the Ablation Trees experiment also in this case a plot is
produced for each one of the tracked variables (i.e. fit time, predict time and
ROC AUC).

##### Example

Since the focus is on `EIF+_EXIFFI+` the only command line argument for this
script is the dataset name:

```bash
./launch_ablation_cont_prediction TEP_ACME
```

>[!warning]
> Note that this experiment cannot be executed on the `PIADE` dataset
> since there are no labels to compute the ROC AUC metric.

#### Ablation Feature Selection

The contamination level not only influences the `AD` model performances but
also the `GFI` scores and thus the feature rankings used to perform the Feature
Selection experiment. For this reason in this ablation study we consider the
relationship between the $AUC_{FS}$ metric and the contamination level.

##### Example

Since the focus is on `EIF+_EXIFFI+` the only command line argument for this
script is the dataset name:

```bash
./launch_ablation_cont_fs TEP_ACME
```

>[!warning]
> Note that this experiment cannot be executed on the `PIADE` dataset
> since there are no labels to compute the average precision metric.
