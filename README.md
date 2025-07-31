# ExIFFI

This is the branch with the code of original ExIFFI that we will use to compare if the C version is equal.

## How to run the experiments

The experiments should produce the following results:

- `GFI` Score Plot → for `ExIFFI,ExIFFI+,DIFFI` interpretation algorithms
- `LFI` Score Plot → for `ACME,KernelSHAP` interpretation algorithms
- Local scoremaps plots → for all interpretation algorithms
- Feature Selection plot → to run on `acquario3`
- Metrics experiment → for the `AD` models

In order to run the experiments we have to consider a pair of `AD` model and interpretation algorithm. The pairs to include in the paper are:

- `IF_DIFFI`
- `EIF+_AcME`
- `EIF+_ExIFFI`
- `EIF+_KernelSHAP`

The experiments can be executed using some bash scripts which are all located in `ExIFFI_Industrial_Test/ExIFFI_original/experiments`

>[!warning]
> There is no guarantee that the scripts will work without errors out of the box. In case there are errors ask me.


### Importance Plots Experiment

The importance plots can be obtained using the `launch_imp_plots.sh` script which produces the `LFI` or `GFI` score plots and, if asked, the local scoremap.

This script has two command line arguments:
- `dataset_name` → use `CoffeData` for the coffe dataset
- `local_scoremaps` → 0 to not plot the local scoremaps, something else to plot them

>[!warning] Add the `CoffeData` to the `Dataset` class
> In order to use the new `CoffeData` dataset this has to be inserted in the `Dataset` class. The dataset should be inserted in `datasets/data/CoffeData`. We can also create two datasets (decide the name you want) to distinguish between `dataset_Tortora` and `dataset_Minato`. Then we probably also need update the `load_data` method inside the `Dataset` class class.

After all this is done we can launch an experiment with `launch_imp_plots`. Inside `launch_imp_plots.sh` we have to define the `model_names` and `interpretations` arrays inserting all the pairs of `AD` models and interpretation models to use. For example if we want to execute the experiment just on `IF_DIFFI` we will do:

```bash
model_names=("IF")
interpretations=("DIFFI")
```

Instead if we want to do that for `IF_DIFFI` and `EIF+_EXIFFI+` we can do

```bash
model_names=("IF" "EIF+")
interpretations=("DIFFI" "EXIFFFI+")
```

and so on

>[!note]
> This thing is not super user friendly, it has to be updated to be more automatic.

Finally to launch the script use the following commands:

Without plotting the scoremaps:

```bash
./launch_imp_plots.sh CoffeData
```

Plotting the scoremaps:

```bash
./launch_imp_plots.sh CoffeData 1
```

### Feature Selection Plot

The script to run to produce the feature selection plots is 

In this case we just have the `dataset_name` command line argument and as usual we have to set the `model_names` and `interpretations` as in the previous script.

So we run simply using:

```bash
./launch_fs_exp.sh CoffeData
```

>[!warning]
> The feature selection experiment may take a while, in particular if there are a lot of features so probably it's better to run it on `acquario2` or `acquario3`.

### Metrics Experiment

For the metrics experiment it's really similar to the feature selection one, so just use:

```bash
./launch_metrics_exp.sh CoffeData
```


