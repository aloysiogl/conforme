# Download data

The data is preprocessed and stored in a pickle file for each dataset. We use git [lfs](https://git-lfs.com/) to store these preprocessed data, and they are available in the data directory of this repo.

To pull the data from git lfs, you should follow the steps [here](https://git-lfs.com/). Then, go to the root of the directory and run:

```bash
git lfs install && git lfs pull
```

In ubuntu I could install git lfs with:

```bash
sudo apt-get install git-lfs
```

## Downloading the data from source

If you wish to download from the original source, you can find the links with instructions in the tables below. We remind you that the following links are subject to change, as well as any external code. For this reason, we provide only high level indications of the steps that we followed. That said, the EEG dataset is probably the easiest one to get working. If you do not have a strong reason we advise you to use the preprocessed data provided in the repository.

| Dataset | Link | Source | Instructions |
|---------|------|--------|--------------|
| EEG     | [link](https://archive.ics.uci.edu/dataset/121/eeg+database)|UCI Machine Learning Repository|Download the data with the link, create a directory `eeg` within the `data` folder in the root of this project, extract the dataset, then extract `SMNI_CMI_TEST` and `SMNI_CMI_TRAIN` and place these two folders inside the newly created `eeg` folder|
| Argoverse     | [LaneGCN link](https://github.com/uber-research/LaneGCN), [Argoverse link](https://www.argoverse.org/av1.html)|LaneGCN repository, Argoverse website|Go to the LaneGCN link clone the repository, train and generate forecasting for the validation data of the Argoverse dataset. We provide a link to Argoverse 1 as well if you do not wish to use the preprocessed data provided by LaneGCN. Place the script `argoverse_processing.py` found in the `data` folder of this repository in the root of the recently cloned LaneGNC and use it to get `aorgoverse_lanegcn.pkl`, use it only for the validation data as the test data does not provide the ground truth. Place the `.pkl` file in the `data` folder.|
| COVID-19     | [link](https://ukhsa-dashboard.data.gov.uk/topics/covid-19?areaType=Lower+Tier+Local+Authority)|UK Health Security Agency|Go to the provided link, and filter the data by area type. Select Lower Tier Local Authority and select csv. You should select the data until May 25, 2021. Then, put the downloaded file `ltla_2021-05-24.csv` directly inside the data directory. Obs.: this source is subject to constant change, and you might not be able to download the data, or you might see different column names, etc. In this case, you would need to adapt the code to deal with any possible changes. The implementation of the data processing is present in `conforme/data_processing/covid.py`|

If the data is placed in the correct folders, with the correct naming in the `data` directory, the code will automatically load it when you run the experiments.
