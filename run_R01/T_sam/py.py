import logging
import os
from typing import Any
import pandas as pd
import numpy as np
import mlflow
import seaborn as sns
from pycaret.clustering import ClusteringExperiment
# Set up the default plotting settings
sns.set_theme(context="paper", style='whitegrid',  palette='deep', font='serif', font_scale=1.7, rc={"figure.dpi": 300})

mlflow.set_tracking_uri('sqlite:///./mlruns.db')
# mlflow.log_dict(_config, mlflow.get_artifact_uri())

# Random State
session_id = 1000
# number of random samples of VM trace collected by the monitoring agents
number_of_random_samples = 100

def clus():

    EP = ClusteringExperiment()

    EP.setup(data=pd.read_csv('trace_dataframe_15000.csv').sample(n=1000), log_experiment=True, pca_components=1,
             pca=True,
             log_plots=['pipeline', 'cluster','tsne','elbow', 'silhouette','distance','distribution'],
             html=False,
             memory='./caching_directory')
    EP.create_model('kmeans')



# __name__
def main():
    clus()
    print('done')
    pass


if __name__ == "__main__":
    main()
