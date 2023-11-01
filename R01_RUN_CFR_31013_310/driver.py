import logging
import os
from typing import Any
import pandas as pd
import numpy as np
import mlflow
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance

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



# model_selection_threshold using the combined score
# model_selection_threshold = 0.90


def create_directories():
    # Create the output directories
    try:
        os.mkdir('out')
    except OSError as error:
        print(error)
    try:
        os.mkdir('out_tbl')
    except OSError as error:
        print(error)
    try:
        os.mkdir('out_figs')
    except OSError as error:
        print(error)
    try:
        os.mkdir('index_encoders')
    except OSError as error:
        print(error)
    try:
        os.mkdir('scores')
    except OSError as error:
        print(error)
    try:
        os.mkdir('models_t1')
    except OSError as error:
        print(error)
    try:
        os.mkdir('models_t2')
    except OSError as error:
        print(error)
    try:
        os.mkdir('setups')
    except OSError as error:
        print(error)
    pass


def load_trace_data(n=None, session_id=session_id):
    '''
    ---------------
    Loading, preprocessing, and returning a dataset from an Azure VM workload trace is the purpose of the load_trace_data function. The first step is to import the required modules and set the location of the dataset file. It loads the data and runs a series of preprocessing operations on it, including encoding categorical features, sampling, feature engineering, timestamp conversion, and feature type logging. It then returns a random subset of the trace_dataframe derived from Azure VM workload trace.
    ---------------
    The load_trace_data function is designed to load, preprocess, and return a dataset from the Azure VM workload trace.
    The dataset is loaded from a CSV file and a random sample of rows can be returned using the n parameter.
    For reproducibility, a random seed (session_id) can be provided.

    The function starts by importing necessary modules and defining the path to the dataset file.
    It then loads the data and applies several preprocessing steps:

    -Encoding categorical features: The function encodes 'VM ID', 'Subscription ID', and 'Deployment ID' columns using the LabelEncoder class from scikit-learn.
    After encoding, it saves the encoders for future use.

    -Sampling: If a number of samples n is provided, the function draws a random sample from the dataset.

    -Feature engineering: The function creates two new features 'Distance 1' and 'Distance 2', calculated as 'MAX CPU' - 'AVG CPU' and '95th Percentile(MAX CPU)' - 'AVG CPU', respectively.
    It also calculates the lifetime of each VM in hours, and creates a binary 'Connected VM' feature that indicates whether a VM's lifetime is equal to the maximum lifetime.

    -Timestamp conversion: The 'Created' and 'Deleted' timestamps are converted to a more readable datetime format.

    -Lastly, the function logs the types of each feature (ignored, categorical, datetime, numeric, ordinal) and returns the preprocessed dataframe along with these feature lists.

    This function is quite useful for load and preprocess Azure VM trace data, making it ready for further analysis or modeling.

    # Return a random sample of trace_dataframe derived from Azure VM workload trace .
    # You can use `random_state` for reproducibility.
    :param n: number of random samples of VM trace collected by the monitoring agents
    :param session_id: a random_state for reproducibility
    :return:
    '''

    # ---- load the required libraries
    # This transformer is used to encode 'VM ID', 'Subscription ID', and 'Deployment ID' VM indexes.
    # Encode target index with value between 0 and n_classes-1.
    from sklearn.preprocessing import LabelEncoder
    # Create portable serialized representations of Python objects
    from pickle import dump

    # ---- load the original data
    print("load the original data")
    data_path = f'/nfs/speed-scratch/m_daragh/datasets/azure2019/original/vmtable.csv.gz'
    headers = ['VM ID', 'Subscription ID', 'Deployment ID', 'Created', 'Deleted', 'MAX CPU', 'AVG CPU',
               '95th Percentile(MAX CPU)', 'VM Type', 'Core Bucket', 'Memory Bucket']
    trace_dataframe = pd.read_csv(data_path, header=None, index_col=False, names=headers, delimiter=',')

    # ---- Encoding the IDs indexes
    print("Encoding the VM IDs")
    VM_ID_scaler = LabelEncoder()
    trace_dataframe['VM ID'] = VM_ID_scaler.fit_transform(trace_dataframe['VM ID'])
    # Save the VM_ID_scaler scaler
    dump(VM_ID_scaler, open('./index_encoders/VM_ID_scaler.pkl', 'wb'))

    print("Encoding the Subscription ID")
    Subscription_ID_le = LabelEncoder()
    trace_dataframe['Subscription ID'] = Subscription_ID_le.fit_transform(trace_dataframe['Subscription ID'])
    # save the Subscription_ID_scaler scaler
    dump(Subscription_ID_le, open('./index_encoders/Subscription_ID_scaler.pkl', 'wb'))

    print("Encoding the Deployment ID")
    Deployment_ID_scaler = LabelEncoder()
    trace_dataframe['Deployment ID'] = Deployment_ID_scaler.fit_transform(trace_dataframe['Deployment ID'])
    # save the Deployment_ID_scaler
    dump(Deployment_ID_scaler, open('./index_encoders/Deployment_ID_scaler.pkl', 'wb'))

    # Return a random sample of items from an axis of object.
    # You can use `random_state` for reproducibility.
    if n is not None:
        print("Get a sample dataset")
        trace_dataframe = trace_dataframe.groupby(by=['VM Type'], group_keys=False).apply(
            lambda x: x.sample(n=n, random_state=session_id))
        trace_dataframe.reset_index(drop=True, inplace=True)

    # ---- Preprocessing trace_dataframe stage one.
    print("Feature engineering")
    print("Original Data", trace_dataframe.shape)

    # 'Compute: \nDistance 1 = MAX CPU - AVG CPU\nDistance 2 = 95th Percentile(MAX CPU) - AVG CPU'
    print('Compute: \nDiff 1 = MAX CPU - AVG CPU\nDiff 2 = 95th Percentile(MAX CPU) - AVG CPU')
    trace_dataframe['Diff 1'] = trace_dataframe['MAX CPU'] - trace_dataframe['AVG CPU']
    trace_dataframe['Diff 2'] = trace_dataframe['95th Percentile(MAX CPU)'] - trace_dataframe['AVG CPU']

    # Compute VM Lifetime based on VM Created and Deleted timestamps and transform to Hour
    print('Compute VM Lifetime based on VM Created and Deleted timestamps and transform to Hour')
    trace_dataframe['Life Time (Hour)'] = np.maximum((trace_dataframe['Deleted'] - trace_dataframe['Created']),
                                                     300) / 3600
    max_life_time = trace_dataframe['Life Time (Hour)'].max()

    # Determine and tage the Connected VMs based on VM Life Time.
    trace_dataframe['Connected VM'] = trace_dataframe['Life Time (Hour)'].apply(
        lambda x: 1 if x == max_life_time else 0)

    # Parse timestamps to datetime
    print('Parse timestamps to datetime')
    ts_start = pd.to_datetime('2019-01-01 00:00:00')
    parse_datetime = True
    if parse_datetime:
        trace_dataframe['Created'] = ts_start + pd.to_timedelta(
            trace_dataframe['Created'], unit='s')
        trace_dataframe['Deleted'] = ts_start + pd.to_timedelta(
            trace_dataframe['Deleted'], unit='s')

    # trace_dataframe schema representation
    schema = dict(ignore_features=['VM ID'],
                  index='VM ID',
                  categorical_features=['Subscription ID', 'Deployment ID', 'VM Type', 'Connected VM'],
                  datetime_index_columns=['Created', 'Deleted'],
                  numeric_features=['MAX CPU', 'AVG CPU', '95th Percentile(MAX CPU)', 'Diff 1', 'Diff 2',
                                    'Life Time (Hour)'],
                  ordinal_features={'Core Bucket': ['2', '4', '8', '24', '>24'],
                                    'Memory Bucket': ['2', '4', '8', '32', '64', '>64']},
                  shape=trace_dataframe.shape)

    print("Dataset schema:\n", schema)
    return trace_dataframe, schema


# Defining main function
def get_best_n_clusters_via_KneeLocator_of_distortion_score(model, X, min_clusters=2, max_clusters=10):
    from yellowbrick.cluster.elbow import distortion_score

    distortions = []
    k_values_ = list(range(min_clusters, max_clusters + 1))
    for n_clusters in k_values_:
        model.set_params(n_clusters=n_clusters)
        model.fit(X)
        distortion = distortion_score(X, model.labels_)  # Sum of squared distances to closest centroid
        distortions.append(distortion)

    # to select the cluster no.
    from yellowbrick.utils import KneeLocator

    kneedle = KneeLocator(k_values_, distortions, curve_nature="convex", curve_direction="decreasing")

    print(f"k: {kneedle.knee}")
    print(f"Knee at {kneedle.knee_y} distortion score")
    return kneedle.knee


def compare_normalization_and_transformation_setups(config, trace_dataframe, index, categorical_features,
                                                    ordinal_features, numeric_features,
                                                    datetime_index_columns):
    '''
    The compare_normalization_and_transformation_setups function compares the effectiveness of different normalization and transformation configurations using different clustering models. The comparison metrics used are Silhouette score (SC), Calinski-Harabasz index (CHI), and Davies-Bouldin index (DBI). The comparison is performed on data prepared by a ClusteringExperiment (a part of the PyCaret library), and the function determines the optimal number of clusters for each clustering model for every configuration.

    Here's a step-by-step explanation:

    For each configuration in config, the function sets up a ClusteringExperiment using the provided data (trace_dataframe), index, and feature definitions. The data is preprocessed with certain transformation and normalization steps, multicollinearity removal, and encoding methods.

    The optimal number of clusters for KMeans and AgglomerativeClustering models is determined using the get_best_n_clusters_via_KneeLocator_of_distortion_score function. This function uses the KneeLocator method to find the "knee point" in the distortion score curve, which is a heuristic way to choose the best number of clusters.

    The clustering models are created and evaluated with the determined number of clusters. The models used include KMeans, AgglomerativeClustering, and MeanShift. For each model, the function records the model details, transformation and normalization methods, and performance metrics (SC, CHI, DBI).

    The models are then saved for later use, and the results are collected into a pandas DataFrame.

    The function then cleans the results and exports the performance metrics table into LaTeX format for each clustering model and saves it as a CSV file.

    The output of the function is a list of dictionaries, where each dictionary contains the model details and performance metrics for a particular configuration.

    This function is an excellent tool for comparing and tuning data preprocessing methods and clustering algorithms. It is designed to work with a diverse set of data types, including categorical, ordinal, numeric, and datetime features. It also takes into account the effects of removing multicollinearity and encoding categorical features on the final clustering performance.

    Here's a brief definition of these parameters:

    config: This is a list of configurations used to normalize and transform the data. Each configuration is a dictionary containing the following key-value pairs:

    'transformation': a boolean indicating whether to perform transformation or not.
    'transformation_method': a string indicating the method of transformation ('yeo-johnson' or 'quantile').
    'normalize': a boolean indicating whether to perform normalization or not.
    'normalize_method': a string indicating the method of normalization ('zscore', 'minmax', 'maxabs', or 'robust').
    trace_dataframe: This is a pandas DataFrame containing the data to be clustered.

    index: This is a string indicating the name of the column to be used as the DataFrame's index.

    categorical_features: This is a list of strings indicating the names of the columns in the DataFrame that should be treated as categorical variables. These features will be encoded using the provided encoding method.

    ordinal_features: This is a list of strings indicating the names of the columns in the DataFrame that should be treated as ordinal variables. These features will be encoded using the provided encoding method.

    numeric_features: This is a list of strings indicating the names of the columns in the DataFrame that should be treated as numerical variables. These features will be normalized and/or transformed as per the configuration.

    datetime_index_columns: This is a list of strings indicating the names of the columns in the DataFrame that contain datetime data. These features will be used to create date-based features such as 'weekday', 'day', 'hour', and 'minute'.

    The function compare_normalization_and_transformation_setups uses these parameters to set up a PyCaret ClusteringExperiment, create clustering models, and evaluate the performance of these models with different normalization and transformation setups.

    The get_best_n_clusters_via_KneeLocator_of_distortion_score function is used within the compare_normalization_and_transformation_setups function. Here is a brief overview of what it does:

    This function identifies the optimal number of clusters to use for a specific clustering model applied to a given dataset. This is accomplished using the distortion score and the KneeLocator utility from the Yellowbrick library.

    The distortion score is a measure of the sum of squared distances from each point to its assigned centroid. As the number of clusters increases, this score generally decreases. However, after a certain number of clusters (the "elbow" or "knee" point), the decrease in distortion score becomes less pronounced. This is often taken as an indication that adding more clusters beyond this point does not provide a substantial improvement in the clustering.

    The KneeLocator utility identifies this "knee" point, which is returned by the function as the recommended number of clusters. The function loops through a range of possible cluster numbers (from min_clusters to max_clusters), computes the distortion score for each, and uses the KneeLocator to identify the optimal number.

    Note that this function specifically uses the Yellowbrick library's implementation of distortion score, and not the "elbow" method sometimes implemented in other libraries (like KMeans in sklearn). The Yellowbrick distortion score is based on the sum of squared distances to the closest centroid, which may be different than other implementations.

    In summary, this function is used in the compare_normalization_and_transformation_setups function to dynamically determine the optimal number of clusters for each configuration of normalization and transformation methods being compared.

    :required functions: get_best_n_clusters_via_KneeLocator_of_distortion_score
    :param config:
    :param trace_dataframe:
    :param index:
    :param categorical_features:
    :param ordinal_features:
    :param numeric_features:
    :param datetime_index_columns:
    :return:
    '''
    from category_encoders import CountEncoder
    from sklearn.cluster import AgglomerativeClustering, KMeans

    # from pycaret.clustering import ClusteringExperiment
    SP = ClusteringExperiment()
    res: list[Any] = []
    for c in config:
        print('\nConfig:', c)
        SP.setup(data=trace_dataframe, index=index,
                 categorical_features=categorical_features,
                 encoding_method=CountEncoder(), max_encoding_ohe=3,
                 ordinal_features=ordinal_features,
                 numeric_features=numeric_features,

                 date_features=datetime_index_columns,
                 create_date_columns=['weekday', 'day', 'hour', 'minute'],
                 # Remove zero variance and perfect remove col-linearity
                 low_variance_threshold=0,  # keep all features with non-zero variance,
                 remove_multicollinearity=True,
                 # Minimum absolute Pearson correlation to identify correlated features. The default value removes equal columns.
                 multicollinearity_threshold=0.99,

                 transformation=True if c['transformation_method'] else False,
                 transformation_method=c['transformation_method'],

                 normalize=True if c['normalize_method'] else False,
                 normalize_method=c['normalize_method'],

                 session_id=session_id,
                 experiment_name=f"Compare Normalization and Transformation Setups",
                 log_experiment=True,
                 log_plots=False,
                 html=False,
                 memory='./caching_directory'
                 )

        SP_kmeans_num_clusters = get_best_n_clusters_via_KneeLocator_of_distortion_score(
            KMeans(random_state=session_id), SP.X_train_transformed)
        SP_hclust_num_clusters = get_best_n_clusters_via_KneeLocator_of_distortion_score(
            AgglomerativeClustering(linkage='ward', affinity='euclidean'), SP.X_train_transformed)

        SP.remove_metric('hs')
        SP.remove_metric('ari')
        SP.remove_metric('cs')

        SP_kmeans = SP.create_model('kmeans', num_clusters=SP_kmeans_num_clusters, round=4)
        res.append(dict(model_index='kmeans',
                        model='KMeans',
                        num_clusters=SP_kmeans_num_clusters,
                        data_pipeline_index=c["data_pipeline_index"],
                        transformation_method=c['transformation_method'],
                        normalize_method=c['normalize_method'],
                        SC=SP.pull()['Silhouette'][0],
                        CHI=SP.pull()['Calinski-Harabasz'][0],
                        DBI=SP.pull()['Davies-Bouldin'][0]))
        SP.save_model(model=SP_kmeans, model_name=f'./models_t1/{c["data_pipeline_index"]}_kmeans', model_only=False)

        SP_hclust = SP.create_model('hclust', linkage='ward', affinity='euclidean', num_clusters=SP_hclust_num_clusters,
                                    round=4)
        res.append(dict(model_index='hclust',
                        model='Agglomerative',
                        num_clusters=SP_hclust_num_clusters,
                        data_pipeline_index=c["data_pipeline_index"],
                        transformation_method=c['transformation_method'],
                        normalize_method=c['normalize_method'],
                        SC=SP.pull()['Silhouette'][0],
                        CHI=SP.pull()['Calinski-Harabasz'][0],
                        DBI=SP.pull()['Davies-Bouldin'][0]))
        SP.save_model(model=SP_hclust, model_name=f'./models_t1/{c["data_pipeline_index"]}_hclust', model_only=False)

        SP_meanshift = SP.create_model('meanshift', round=4)
        res.append(dict(model_index='meanshift',
                        model='MeanShift',
                        num_clusters=np.unique(SP_meanshift.labels_).size,
                        data_pipeline_index=c["data_pipeline_index"],
                        transformation_method=c['transformation_method'],
                        normalize_method=c['normalize_method'],
                        SC=SP.pull()['Silhouette'][0],
                        CHI=SP.pull()['Calinski-Harabasz'][0],
                        DBI=SP.pull()['Davies-Bouldin'][0]))
        SP.save_model(model=SP_meanshift, model_name=f'./models_t1/{c["data_pipeline_index"]}_meanshift', model_only=False)


    # clean the results and export the tables
    res_df = pd.DataFrame(res)
    res_df_ = res_df[['model', 'data_pipeline_index', 'num_clusters', 'SC', 'CHI', 'DBI']]
    res_df_.columns = ['Model', 'Data Pipeline', 'Clusters', 'SC', 'CHI', 'DBI']
    res_df_.set_index('Model', inplace=True)
    # res_df_.sort_values(by=['Model', 'SC'], ascending=False, inplace=True)
    for index in np.unique(res_df_.index):
        res_df__ = res_df_.loc[index]
        res_df__.reset_index(inplace=True, drop=True)
        res_df__.to_latex(buf=f'out_tbl/SP_norm_tran_Results_{index}.tex',
                          caption=f'Comparison of Various Normalization and Transformation Methods (via SP, {index})',
                          label=f'tbl:Comparison_SP_{index}_norm_tran', position='t',
                          index=False,escape=False)
    res_df.to_csv(f'./scores/Comparison_SP_norm_tran.csv')
    return res


def compare_normalization_transformation_pca_setups(config, trace_dataframe, index, categorical_features,
                                                    ordinal_features, numeric_features,
                                                    datetime_index_columns):
    '''
    The function compare_normalization_transformation_pca_setups is similar to the compare_normalization_and_transformation_setups function but with an additional parameter to evaluate: the number of PCA components. Principal Component Analysis (PCA) is a technique often used for dimensionality reduction before performing machine learning tasks.

    The key parameters of the function are:

    config: This is a list of dictionaries where each dictionary represents a different configuration of parameters to use in the experiment. This should include 'normalize_method', 'transformation_method', and 'pca_components' for each configuration.

    trace_dataframe: This is the dataframe which contains the data on which to perform clustering.

    index, categorical_features, ordinal_features, numeric_features, datetime_index_columns: These parameters specify the nature of the features present in the trace_dataframe, such as categorical features, numeric features, etc.

    The function goes through each configuration in the config list and performs the following steps:

    Sets up the clustering experiment using the provided configuration and the setup function from PyCaret's ClusteringExperiment module.
    Determines the best number of clusters for KMeans and Agglomerative Clustering using the get_best_n_clusters_via_KneeLocator_of_distortion_score function.
    Removes some default metrics ('hs', 'ari', 'cs') from the experiment.
    Creates and evaluates three different models: KMeans, Agglomerative Clustering, and MeanShift.
    Appends the evaluation metrics and parameters for each model into the results list.
    Exports the final results list into a CSV file and a series of LaTeX tables, one for each clustering model.
    This function allows you to compare the effectiveness of different configurations of normalization methods, transformation methods, and the number of PCA components for three different clustering models (KMeans, Agglomerative Clustering, and MeanShift). The function uses various performance metrics like Silhouette Score (SC), Calinski-Harabasz Index (CHI), and Davies-Bouldin Index (DBI) to evaluate the performance of each configuration.

    :param config:
    :param trace_dataframe:
    :param index:
    :param categorical_features:
    :param ordinal_features:
    :param numeric_features:
    :param datetime_index_columns:
    :return:
    '''
    from category_encoders import CountEncoder
    from sklearn.cluster import AgglomerativeClustering, KMeans

    # from pycaret.clustering import ClusteringExperiment
    SP = ClusteringExperiment()

    res: list[Any] = []
    for c in config:
        print('\nConfig:', c)
        SP.setup(data=trace_dataframe, index=index,
                 categorical_features=categorical_features,
                 encoding_method=CountEncoder(), max_encoding_ohe=3,
                 ordinal_features=ordinal_features,
                 numeric_features=numeric_features,

                 date_features=datetime_index_columns,
                 create_date_columns=['weekday', 'day', 'hour', 'minute'],
                 # Remove zero variance and perfect remove col-linearity
                 low_variance_threshold=0,  # keep all features with non-zero variance,
                 remove_multicollinearity=True,
                 # Minimum absolute Pearson correlation to identify correlated features. The default value removes equal columns.
                 multicollinearity_threshold=0.99,

                 pca=True,
                 pca_method='linear',
                 pca_components=c['pca_components'],

                 transformation=True if c['transformation_method'] else False,
                 transformation_method=c['transformation_method'],

                 normalize=True if c['normalize_method'] else False,
                 normalize_method=c['normalize_method'],

                 session_id=session_id,
                 experiment_name=f"Compare Normalization, Transformation, and PCA Setups",
                 log_experiment=True,
                 log_plots=False,
                 html=False,
                 memory='./caching_directory'
                 )

        SP_kmeans_num_clusters = get_best_n_clusters_via_KneeLocator_of_distortion_score(
            KMeans(random_state=session_id), SP.X_train_transformed)
        SP_hclust_num_clusters = get_best_n_clusters_via_KneeLocator_of_distortion_score(
            AgglomerativeClustering(linkage='ward', affinity='euclidean'), SP.X_train_transformed)

        SP.remove_metric('hs')
        SP.remove_metric('ari')
        SP.remove_metric('cs')

        SP_kmeans = SP.create_model('kmeans', num_clusters=SP_kmeans_num_clusters, round=4)
        res.append(dict(model_index='kmeans',
                        model='KMeans',
                        num_clusters=SP_kmeans_num_clusters,
                        data_pipeline_index=c["data_pipeline_index"],
                        transformation_method=c['transformation_method'],
                        normalize_method=c['normalize_method'],
                        pca_components=c['pca_components'],
                        cumsum_EV=c['cumsum_EV'],
                        SC=SP.pull()['Silhouette'][0],
                        CHI=SP.pull()['Calinski-Harabasz'][0],
                        DBI=SP.pull()['Davies-Bouldin'][0]))
        SP.save_model(model=SP_kmeans, model_name=f'./models_t2/{c["data_pipeline_index"]}_kmeans', model_only=False)

        SP_hclust = SP.create_model('hclust', linkage='ward', affinity='euclidean', num_clusters=SP_hclust_num_clusters,
                                    round=4)
        res.append(dict(model_index='hclust',
                        model='Agglomerative',
                        num_clusters=SP_hclust_num_clusters,
                        data_pipeline_index=c["data_pipeline_index"],
                        transformation_method=c['transformation_method'],
                        normalize_method=c['normalize_method'],
                        pca_components=c['pca_components'],
                        cumsum_EV=c['cumsum_EV'],
                        SC=SP.pull()['Silhouette'][0],
                        CHI=SP.pull()['Calinski-Harabasz'][0],
                        DBI=SP.pull()['Davies-Bouldin'][0]))
        SP.save_model(model=SP_hclust, model_name=f'./models_t2/{c["data_pipeline_index"]}_hclust', model_only=False)

        SP_meanshift = SP.create_model('meanshift', round=4)
        res.append(dict(model_index='meanshift',
                        model='MeanShift',
                        num_clusters=np.unique(SP_meanshift.labels_).size,
                        data_pipeline_index=c["data_pipeline_index"],
                        transformation_method=c['transformation_method'],
                        normalize_method=c['normalize_method'],
                        pca_components=c['pca_components'],
                        cumsum_EV=c['cumsum_EV'],
                        SC=SP.pull()['Silhouette'][0],
                        CHI=SP.pull()['Calinski-Harabasz'][0],
                        DBI=SP.pull()['Davies-Bouldin'][0]))
        SP.save_model(model=SP_meanshift, model_name=f'./models_t2/{c["data_pipeline_index"]}_meanshift', model_only=False)


    # clean the results and export the tables
    res_df = pd.DataFrame(res)
    res_df_ = res_df[['model', 'data_pipeline_index', 'num_clusters', 'SC', 'CHI', 'DBI']]
    res_df_.columns = ['Model', 'Data Pipeline','Clusters', 'SC', 'CHI', 'DBI']
    res_df_.set_index('Model', inplace=True)
    # res_df_.sort_values(by=['Model', 'SC'], ascending=False, inplace=True)
    for index in np.unique(res_df_.index):
        res_df__ = res_df_.loc[index]
        res_df__.reset_index(inplace=True, drop=True)
        res_df__.to_latex(buf=f'out_tbl/SP_pca_norm_tran_Results_{index}.tex',
                          caption=f'Comparison of Various Normalization and Transformation Methods (via SP_PCA, {index})',
                          label=f'tbl:Comparison_SP_PCA_{index}_norm_tran', position='t',
                          index=False, escape=False)
    res_df.to_csv(f'./scores/Comparison_SP_PCA_norm_tran.csv')
    return res


def get_pca_components_setups(config, trace_dataframe, index, categorical_features, ordinal_features, numeric_features,
                              datetime_index_columns):
    '''
    The get_pca_components_setups function creates different setups for PCA (Principal Component Analysis), a dimensionality reduction technique. It operates over a given list of configurations (defined by config), and for each configuration, it prepares the data, applies PCA, and determines the optimal number of principal components. The function returns a list of results that include the transformation method, normalization method, optimal number of PCA components, and cumulative explained variance for each configuration.

    Here's a more detailed explanation of how the function works:

    Import necessary modules and create a ClusteringExperiment object, SP.

    For each configuration in the config list, it sets up the experiment using the trace_dataframe as the dataset, with the provided index, categorical features, ordinal features, numeric features, and datetime index columns. It also applies certain data preparation steps, including feature encoding, creation of date-related features, transformation and normalization (specified by the configuration), and PCA.

    With the prepared data, it then calculates the optimal number of PCA components by using the get_best_pca_components_via_KneeLocator_of_cumsum_explained_variance function on the transformed dataset SP.X_transformed.

    It records the transformation method, normalization method, optimal number of PCA components, and cumulative explained variance for the configuration and appends the result to the res list.

    Finally, it converts the list of results into a pandas DataFrame and saves it as a LaTeX file and a CSV file. The DataFrame includes columns for transformation method, normalization method, optimal number of PCA components, and cumulative explained variance for each configuration.

    Please note that get_best_pca_components_via_KneeLocator_of_cumsum_explained_variance is not defined in this code. It is assumed that this function uses Knee Locator to determine the optimal number of PCA components where the increase in cumulative explained variance drops off significantly.

    This function is highly useful for comparing different data preparation methods and their effects on dimensionality reduction and variability explanation in PCA. It's a part of a pipeline for evaluating and tuning different machine learning models.
    :param config:
    :param trace_dataframe:
    :param index:
    :param categorical_features:
    :param ordinal_features:
    :param numeric_features:
    :param datetime_index_columns:
    :return:
    '''
    from category_encoders import CountEncoder

    # from pycaret.clustering import ClusteringExperiment
    SP = ClusteringExperiment()

    res: list[Any] = []
    for c in config:
        SP.setup(data=trace_dataframe, index=index,
                 categorical_features=categorical_features,
                 encoding_method=CountEncoder(), max_encoding_ohe=3,
                 ordinal_features=ordinal_features,
                 numeric_features=numeric_features,

                 date_features=datetime_index_columns,
                 create_date_columns=['weekday', 'day', 'hour', 'minute'],

                 transformation=True if c['transformation_method'] else False,
                 transformation_method=c['transformation_method'],

                 normalize=True if c['normalize_method'] else False,
                 normalize_method=c['normalize_method'],

                 session_id=session_id,
                 experiment_name=f"Prepare PCA Setups",
                 log_experiment=True,
                 log_plots=False,
                 html=False,
                 memory='./caching_directory'
                 )
        pca_components, cumsum_EV = get_best_pca_components_via_KneeLocator_of_cumsum_explained_variance(
            SP.X_transformed)
        res.append(dict(data_pipeline_index=c["data_pipeline_index"],
                        transformation_method=c['transformation_method'],
                        normalize_method=c['normalize_method'],
                        pca_components=pca_components,
                        cumsum_EV=cumsum_EV))

    # PCA components and cumulative explained variance (CEV) for different combinations of transformation and normalization methods
    res_df = pd.DataFrame(res)

    res_df.columns = ['Data Pipeline','Transformation', 'Normalization', 'PCA Components', 'CEV']

    res_df.to_latex(buf=f'out_tbl/SP_PCA_CEV_transformation_normalization_setups.tex',
                    caption=f'PCA components and CEV for various combinations of transformation and normalization methods',
                    label=f'tbl:SP_PCA_CEV_norm_tran_setups', position='t', index=False, escape=False)

    res_df.to_csv(f'./setups/SP_PCA_CEV_transformation_normalization_setups.csv', index=False)
    return res


def get_normalization_and_transformation_setups():
    '''
    The get_normalization_and_transformation_setups function generates a list of different configuration dictionaries, where each dictionary contains specific settings for data normalization and transformation.

    The function first defines lists of possible methods for normalization and transformation, as well as boolean options to decide whether to apply normalization or transformation.

    The function then iterates over the possible combinations of normalization and transformation options and constructs a configuration dictionary for each combination.

    For each transformation option, if transformation is set to True, it will create a configuration for each transformation method combined with each normalization option. If the normalization option is set to True, it will create a configuration for each normalization method. If normalization is set to False, a configuration will still be created, but with no normalization method specified.

    If the transformation option is set to False, configurations are created for each normalization option in a similar fashion.

    In the end, the function returns a list of all possible configurations. This function can be used for systematically exploring different normalization and transformation settings during data preprocessing.

    This kind of methodical setup allows us to experiment with various combinations of transformations and normalization techniques, which can significantly help in tasks such as model selection and hyperparameter tuning.

    :return: normalization_and_transformation_setups
    '''
    normalize_methods = ['zscore', 'minmax', 'maxabs', 'robust']
    transformation_methods = ['yeo-johnson', 'quantile']
    normalize = [False, True]
    transformation = [False, True]

    config = []
    for tran in transformation:
        cp = dict()
        if tran:
            cp['transformation'] = True
            for transformation_method in transformation_methods:
                cp['transformation_method'] = transformation_method
                for norm in normalize:
                    if norm:
                        cp['normalize'] = True
                        for normalize_method in normalize_methods:
                            cp['normalize_method'] = normalize_method
                            # print(cp)
                            config.append(cp.copy())
                    else:
                        cp['normalize'] = False
                        cp['normalize_method'] = None
                        # print(cp)
                        config.append(cp.copy())
        else:
            cp['transformation'] = False
            cp['transformation_method'] = None
            for norm in normalize:
                if norm:
                    cp['normalize'] = True
                    for normalize_method in normalize_methods:
                        cp['normalize_method'] = normalize_method
                        # print(cp)
                        config.append(cp.copy())
                else:
                    cp['normalize'] = False
                    cp['normalize_method'] = None
                    # print(cp)
                    config.append(cp.copy())

    # Adding Data pipeline index in Latex format
    for index, cp in zip(range(len(config)), config):
        cp['data_pipeline_index'] = f"$P_{{{str(index + 1)}}}$"

    return config


def get_best_pca_components_via_KneeLocator_of_cumsum_explained_variance(data):
    import numpy as np
    # import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Perform PCA
    pca = PCA(n_components=data.shape[1])  # Set the maximum number of components to the original dimension
    pca.fit(data)

    # Calculate the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # to select the pca_components
    from yellowbrick.utils import KneeLocator

    kneedle = KneeLocator(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio,
                          curve_nature="concave", curve_direction="increasing")

    print(f"pca components: {kneedle.knee}")
    print(f"Knee at {np.round(kneedle.knee_y, 2)} cumulative sum of explained variance")
    return kneedle.knee, np.round(kneedle.knee_y, 2)


def calculate_combine_score_and_select_models_based_on_median(scores_df, weight_silhouette=0.34,
                                                              weight_calinski_harabasz=0.33,
                                                              weight_davies_bouldin=0.33):
    '''
    The function calculate_combine_score_and_selecte_models_based_on_median is designed to combine multiple clustering metrics and select the model configurations that are above the median of the combined score.

    The function takes four parameters:

    scores_df: This is a dataframe that contains the Silhouette Score (SC), Calinski-Harabasz Index (CHI), and Davies-Bouldin Index (DBI) for each model configuration.

    weight_silhouette: This is the weight assigned to the Silhouette Score when calculating the combined score. The default value is 0.34.

    weight_calinski_harabasz: This is the weight assigned to the Calinski-Harabasz Index when calculating the combined score. The default value is 0.33.

    weight_davies_bouldin: This is the weight assigned to the Davies-Bouldin Index when calculating the combined score. The default value is 0.33.

    The function works as follows:

    First, it normalizes the Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index to a common range of 0 to 1 using sklearn's MinMaxScaler. For the Davies-Bouldin Index, which is a metric where lower values are better, the function also inverts the score so that higher values are better.

    Then, it calculates a combined score for each model configuration by taking a weighted average of the normalized scores, using the weights provided as function parameters.

    Finally, it selects the model configurations whose combined score is above the median of all combined scores.

    This function is useful for choosing the best model configurations based on multiple clustering metrics. By normalizing the scores and calculating a weighted average, it allows you to compare configurations on a fair and balanced basis.

    :param scores_df:
    :param weight_silhouette:
    :param weight_calinski_harabasz:
    :param weight_davies_bouldin:
    :return:
    '''
    # Define the weights for each metric (you can adjust the weights based on your preference)
    from pickle import dump
    # Normalize the scores to a common range form 0 to 1 using min-max scaler
    from sklearn.preprocessing import MinMaxScaler
    silhouette_scaler = MinMaxScaler(feature_range=(0, 1))
    scores_df['normalized_SC'] = silhouette_scaler.fit_transform(scores_df[['SC']])
    # scores_df['normalized2_SC'] = (scores_df[['SC']] + 1) / 2 # considering the original range
    dump(silhouette_scaler, open('./scores/silhouette_scaler.pkl', 'wb'))

    calinski_harabasz_scaler = MinMaxScaler(feature_range=(0, 1))
    scores_df['normalized_CHI'] = calinski_harabasz_scaler.fit_transform(scores_df[['CHI']])
    dump(calinski_harabasz_scaler, open('./scores/calinski_harabasz_scaler.pkl', 'wb'))

    davies_bouldin_scaler = MinMaxScaler(feature_range=(0, 1))
    # 1 - normalized DBI to change the optimal values from lower to higher
    scores_df['normalized_DBI'] = 1 - davies_bouldin_scaler.fit_transform(scores_df[['DBI']])
    dump(davies_bouldin_scaler, open('./scores/davies_bouldin_scaler.pkl', 'wb'))

    # Combine the scores using weighted average
    scores_df['combined_score'] = (weight_silhouette * scores_df['normalized_SC'] +
                                   weight_calinski_harabasz * scores_df['normalized_CHI'] +
                                   weight_davies_bouldin * scores_df['normalized_DBI'])

    selection_threshold = scores_df['combined_score'].median(axis=0)

    selected_CP = scores_df[scores_df['combined_score'] > selection_threshold]

    res_df_ = selected_CP[
        ['data_pipeline_index', 'model', 'num_clusters', 'normalized_SC', 'normalized_CHI', 'normalized_DBI',
         'combined_score']]
    res_df_.columns = ['Data Pipeline', 'Base Model', 'Clusters', 'normalized_SC', 'normalized_CHI', 'normalized_DBI',
                       'combined_score']
    res_df_.sort_values(by=['combined_score'], ascending=False, inplace=True)

    res_df_['normalized_SC'] = res_df_['normalized_SC'].mul(100).round(2).astype(str).add(' \%')
    res_df_['normalized_CHI'] = res_df_['normalized_CHI'].mul(100).round(2).astype(str).add(' \%')
    res_df_['normalized_DBI'] = res_df_['normalized_DBI'].mul(100).round(2).astype(str).add(' \%')
    res_df_['combined_score'] = res_df_['combined_score'].mul(100).round(2).astype(str).add(' \%')

    # for index in np.unique(res_df_.index):
    # res_df__ = res_df_.loc[index]
    res_df_.reset_index(inplace=True, drop=True)
    res_df_.to_latex(buf=f'out_tbl/SP_PCA_Combined_Score_Results.tex',
                     caption=f'The selected clustering pipelines sorted based on combined score',
                     label=f'tbl:SP_PCA_Combined_Score_Results', position='t',
                     index=False, escape=False)

    return selected_CP


def assigned_clustering(trace_dataframe, index, categorical_features, ordinal_features, numeric_features,
                        datetime_index_columns, selected_models_list):
    '''
    The function assigned_clustering is designed to assign clustering labels to the data using the selected model configurations.

    The function takes seven parameters:

    trace_dataframe: The original input data as a pandas DataFrame.
    index: The index for the data.
    categorical_features: The categorical features in the data.
    ordinal_features: The ordinal features in the data.
    numeric_features: The numeric features in the data.
    datetime_index_columns: The date features in the data.
    selected_models_list: The selected model configurations as a list of dictionaries, where each dictionary represents a model configuration.
    The function works as follows:

    First, it initializes a ClusteringExperiment object from PyCaret and sets up the data with the specified features and index.

    Then, it loops over the selected model configurations. For each configuration, it loads the model from a previously saved file, assigns clustering labels to the data using the model, and merges the labels with the original data.

    Finally, it returns the data with the assigned clustering labels and a list of the column names for the clustering labels.

    This function is useful for assigning clustering labels to data based on multiple model configurations, which can help you understand the clustering structure of the data from different perspectives.
    :param trace_dataframe:
    :param index:
    :param categorical_features:
    :param ordinal_features:
    :param numeric_features:
    :param datetime_index_columns:
    :param selected_models_list:
    :return:
    '''
    from category_encoders import CountEncoder

    # from pycaret.clustering import ClusteringExperiment

    SP = ClusteringExperiment()

    SP.setup(data=trace_dataframe, index=index,
             categorical_features=categorical_features,
             encoding_method=CountEncoder(), max_encoding_ohe=3,
             ordinal_features=ordinal_features,
             numeric_features=numeric_features,

             date_features=datetime_index_columns,
             create_date_columns=['weekday', 'day', 'hour', 'minute'],

             session_id=session_id,
             experiment_name=f"load and assign clustering",
             log_experiment=True,
             log_plots=False,
             html=False,
             memory='./caching_directory'
             )

    selected_models_cols = []
    data_clustered = SP.X
    for m in selected_models_list.to_dict('records'):
        print(m['data_pipeline_index'], m['model_index'])
        mo = SP.load_model(f"models_t2/{m['data_pipeline_index']}_{m['model_index']}")
        dd = SP.assign_model(mo)
        dd.rename(columns={'Cluster': f"{m['data_pipeline_index']}_{m['model_index']}"}, inplace=True)
        selected_models_cols.append(f"{m['data_pipeline_index']}_{m['model_index']}")
        data_clustered = data_clustered.merge(dd)

    return data_clustered, selected_models_cols


def ensemble_clustering(Cluster_labels, meta_clustering_model_index, pca,  setup_kwargs,
                        meta_model_kwargs,
                        session_id):
    '''
    create_coassociation_matrix() function will create a matrix where each entry [i, j] represents how many times data points i and j appear in the same cluster across the different clustering models.

    The ensemble_clustering() function will use this co-association matrix as input for a meta clustering algorithm. This is a common approach in ensemble clustering techniques. The rationale is that data points that often end up in the same cluster probably have some underlying similarity, even if individual clustering algorithms disagree on their exact clustering.

    This function first checks whether the co-association matrix should be used and if PCA (Principal Component Analysis) should be applied for dimensionality reduction. If used_coassoc_matrix is True, the function creates the co-association matrix and uses it as data input.

    If pca is also True, the function estimates the best number of PCA components and applies PCA transformation to the data before setting up the PyCaret ClusteringExperiment. If pca is False, it sets up the ClusteringExperiment without PCA.

    Next, the function removes some default metrics from PyCaret's ClusteringExperiment, presumably because they are not relevant to this analysis.

    The function then checks the meta_clustering_model_index to decide which meta clustering algorithm to apply. If the meta clustering model is either K-means or Agglomerative Clustering, the function uses the distortion score or the linkage criterion respectively to determine the optimal number of clusters.

    Finally, the function creates and fits the meta clustering model using the provided hyperparameters, assigns the resulting cluster labels to the data, and returns the fitted model and the labels.

    This ensemble clustering approach takes advantage of the strengths of different clustering algorithms and can potentially yield more robust clustering results.

    Overall, this is a sophisticated and well-thought-out approach to cluster analysis. The function is complex and has a lot of steps, so thorough testing is essential to ensure that it works as expected.

    :param labels:
    :param meta_clustering_model_index:
    :param pca:
    :param used_coassoc_matrix:
    :param setup_kwargs:
    :param meta_model_kwargs:
    :param session_id:
    :return:
    '''



    EP = ClusteringExperiment()
    pca_components = ""
    cumsum_EV = ""
    if pca:
        EP.setup(data=Cluster_labels, **setup_kwargs)
        pca_components, cumsum_EV = get_best_pca_components_via_KneeLocator_of_cumsum_explained_variance(EP.train_transformed)

        EP.setup(data=Cluster_labels, pca=True, pca_components=pca_components, session_id=session_id, **setup_kwargs)
        mlflow.log_param('cumsum_EV', cumsum_EV)
    else:
        EP.setup(data=Cluster_labels, **setup_kwargs)

    mlflow.log_param('meta_clustering_model_index', meta_clustering_model_index)

    EP.remove_metric('hs')
    EP.remove_metric('ari')
    EP.remove_metric('cs')
    meta_clustering_model_name ="-"
    if meta_clustering_model_index == 'meanshift':
        meta_clustering_model_name = 'MeanShift'
        from sklearn.cluster import MeanShift
        title = f"Model = ({meta_clustering_model_name}),     PCA = ({pca})"
        # Instantiate the clustering model and InterclusterDistance
        visualizer = InterclusterDistance(MeanShift(n_jobs=-1), legend=False, random_state=session_id,
                                          title=title)
        visualizer.fit(EP.X_train_transformed)  # Fit the data to the visualizer
        visualizer.show(outpath=f'out_figs/InterclusterDistance_{meta_clustering_model_index}_{pca}.png',
                        clear_figure=True)  # Finalize and render the figure

    if meta_clustering_model_index in ['kmeans', 'hclust', 'sc', 'birch', 'kmodes']:
        from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, Birch
        from kmodes.kmodes import KModes
        if meta_clustering_model_index == 'kmeans':
            meta_clustering_model_name = 'KMeans'
            # num_clusters = get_best_n_clusters_via_KneeLocator_of_distortion_score(
            # KMeans(random_state=session_id), EP.X_train_transformed)

            title = f"Model = ({meta_clustering_model_name}),     PCA = ({pca})"
            visualizer = KElbowVisualizer(KMeans(random_state=session_id), k=(2, 10), timings=False, metric="distortion",
                                          title=title)
            visualizer.fit(EP.X_train_transformed)  # Fit the data to the visualizer
            # print("num_clusters", num_clusters)
            # print(visualizer.k, "visualizer.k_scores_", visualizer.knee_value)
            visualizer.show(outpath=f'out_figs/KElbow_{meta_clustering_model_index}_{pca}.png',
                            clear_figure=True)  # Finalize and render the figure
            num_clusters = visualizer.elbow_value_
            meta_model_kwargs['num_clusters'] = num_clusters

            # Instantiate the clustering model and InterclusterDistance
            visualizer = InterclusterDistance(KMeans(random_state=session_id, n_clusters=num_clusters), legend=False, random_state=session_id,
                                              title=title)
            visualizer.fit(EP.X_train_transformed)  # Fit the data to the visualizer
            visualizer.show(outpath=f'out_figs/InterclusterDistance_{meta_clustering_model_index}_{pca}.png',
                            clear_figure=True)  # Finalize and render the figure

            # Instantiate the clustering model and visualizer
            visualizer = SilhouetteVisualizer(KMeans(random_state=session_id, n_clusters=num_clusters), colors='yellowbrick', title=title)
            visualizer.fit(EP.X_train_transformed)  # Fit the data to the visualizer
            visualizer.show(outpath=f'out_figs/Silhouette_{meta_clustering_model_index}_{pca}.png', clear_figure=True)

        if meta_clustering_model_index == 'hclust':
            meta_clustering_model_name = 'Agglomerative'
            # num_clusters = get_best_n_clusters_via_KneeLocator_of_distortion_score(
            #     AgglomerativeClustering(linkage='ward', affinity='euclidean'), EP.X_train_transformed)

            title = f"Model = ({meta_clustering_model_name}),     PCA = ({pca})"
            visualizer = KElbowVisualizer(AgglomerativeClustering(linkage='ward', affinity='euclidean'), k=(2, 10), timings=False, metric="distortion",
                                          title=title)
            visualizer.fit(EP.X_train_transformed)  # Fit the data to the visualizer
            # print("num_clusters", num_clusters)
            # print(visualizer.k, "visualizer.k_scores_", visualizer.knee_value)
            visualizer.show(outpath=f'out_figs/KElbow_{meta_clustering_model_index}_{pca}.png',
                            clear_figure=True)  # Finalize and render the figure
            num_clusters = visualizer.elbow_value_

            meta_model_kwargs['num_clusters'] = num_clusters

        if meta_clustering_model_index == 'sc':
            meta_clustering_model_name = 'Spectral'
            num_clusters = get_best_n_clusters_via_KneeLocator_of_distortion_score(
                SpectralClustering(), EP.X_train_transformed)
            meta_model_kwargs['num_clusters'] = num_clusters

        if meta_clustering_model_index == 'birch':
            meta_clustering_model_name = 'Birch'
            num_clusters = get_best_n_clusters_via_KneeLocator_of_distortion_score(
                Birch(), EP.X_train_transformed)
            meta_model_kwargs['num_clusters'] = num_clusters

        if meta_clustering_model_index == 'kmodes':
            meta_clustering_model_name = 'KModes'
            num_clusters = get_best_n_clusters_via_KneeLocator_of_distortion_score(
                KModes(random_state=session_id), EP.X_train_transformed)
            meta_model_kwargs['num_clusters'] = num_clusters

    meta_clustering_model = EP.create_model(meta_clustering_model_index, round=4, **meta_model_kwargs)
    mlflow.log_param('pca', pca)
    ensemble_labels = EP.assign_model(meta_clustering_model)['Cluster']

    res = dict(meta_clustering_model_index=meta_clustering_model_index,
               meta_clustering_model_name=meta_clustering_model_name,
               num_clusters=ensemble_labels.unique().__len__(),
               pca=pca,
               pca_components=pca_components,
               cumsum_EV = cumsum_EV,
               SC=EP.pull()['Silhouette'][0],
               CHI=EP.pull()['Calinski-Harabasz'][0],
               DBI=EP.pull()['Davies-Bouldin'][0])

    ensemble_labels.name = f'Ensemble_Cluster_{meta_clustering_model_index}_{pca}'
    return meta_clustering_model, ensemble_labels, res


def main():
    global Cluster_labels_coassociation_matrix
    import time
    '''
    main() function is an overarching function that controls the flow of the entire process of performing ensemble clustering on a dataset from the Azure VM workload trace. It makes use of several utility functions along the way to preprocess and transform the data, apply clustering algorithms, and save the results.

    Below, I will discuss each part of this main function:

        Creating Directories: The function create_directories() is used at the beginning. Presumably, this function creates necessary directories for storing outputs or intermediate results.

        Loading and Preprocessing Trace Data: The function load_trace_data() is used to load the dataset from the Azure VM workload trace, preprocess it, and return different components of the dataset such as the index, categorical features, ordinal features, numeric features, and datetime index columns.

        Generating Normalization and Transformation Setups: The function get_normalization_and_transformation_setups() generates a list of configuration dictionaries for data normalization and transformation.

        Generating PCA Components Setups: The function get_pca_components_setups() generates different setups for Principal Component Analysis (PCA), which is used for dimensionality reduction.

        Comparing Normalization and Transformation Setups: The function compare_normalization_and_transformation_setups() compares the effectiveness of different normalization and transformation configurations using different clustering models.

        Comparing Normalization, Transformation, and PCA Setups: The function compare_normalization_transformation_pca_setups() extends the previous comparison by including the number of PCA components as an additional parameter to evaluate.

        Calculating Combined Score and Selecting Models: The function calculate_combine_score_and_selecte_models_based_on_median() combines multiple clustering metrics and selects the model configurations that are above the median of the combined score.

        Assigning Clustering: The function assigned_clustering() assigns clustering labels to the data using the selected model configurations.

        Ensemble Clustering: Ensemble clustering of selected model configurations pipelines using their cluster labels. The function ensemble_clustering() is used here, which uses a selected meta clustering model to cluster the labels. The function creates a co-association matrix from the labels if required and performs PCA if specified.

        Saving Results: After the ensemble clustering, the function merges the resulting ensemble labels with the original trace dataframe and saves the result to a CSV file.

        Finally, the function prints "Done" to signal that all steps have been completed successfully.
    :return:
    '''
    #     Creating Directories: The function create_directories() is used at the beginning.
    #     Presumably, this function creates necessary directories for storing outputs or intermediate results.
    create_directories()

    #
    #     The load_trace_data function is designed to load, preprocess, and return a dataset from the Azure VM workload trace.
    #     The dataset is loaded from a CSV file and a random sample of rows can be returned using the n parameter.
    #     For reproducibility, a random seed (session_id) can be provided.
    # trace_dataframe, schema = load_trace_data(n=number_of_random_samples, session_id=session_id)

    trace_dataframe = pd.read_csv('./trace_dataframe_15000.csv',index_col=False)
    trace_dataframe['Created'] = pd.to_datetime(trace_dataframe['Created'])
    trace_dataframe['Deleted'] = pd.to_datetime(trace_dataframe['Deleted'])
    schema = {'ignore_features': ['VM ID'],
              'index': 'VM ID',
              'categorical_features': ['Subscription ID', 'Deployment ID', 'VM Type', 'Connected VM'],
              'datetime_index_columns': ['Created', 'Deleted'],
              'numeric_features': ['MAX CPU', 'AVG CPU', '95th Percentile(MAX CPU)', 'Diff 1', 'Diff 2', 'Life Time (Hour)'],
              'ordinal_features': {'Core Bucket': ['2', '4', '8', '24', '>24'],
                                   'Memory Bucket': ['2', '4', '8', '32', '64', '>64']},
              'shape': (15000, 15)}
    # trace_dataframe = trace_dataframe.sample(n=100, random_state=session_id, ignore_index=True)

    #
    #   The get_normalization_and_transformation_setups function generates a list of different configuration dictionaries,
    # where each dictionary contains specific settings for data normalization and transformation.
    normalization_and_transformation_setups = get_normalization_and_transformation_setups()

    #   The get_pca_components_setups function creates different setups for PCA (Principal Component Analysis),
    # a dimensionality reduction technique. It operates over a given list of configurations (defined by config),
    # and for each configuration, it prepares the data, applies PCA, and determines the optimal number of principal components.
    # The function returns a list of results that include the transformation method, normalization method, optimal number
    # of PCA components, and cumulative explained variance for each configuration.
    pca_components_setups = get_pca_components_setups(normalization_and_transformation_setups,
                                                      trace_dataframe,
                                                      schema['index'],
                                                      schema['categorical_features'],
                                                      schema['ordinal_features'],
                                                      schema['numeric_features'],
                                                      schema['datetime_index_columns'])

    # The compare_normalization_and_transformation_setups function compares the effectiveness of different
    # normalization and transformation configurations using different clustering models.
    # The comparison metrics used are Silhouette score (SC), Calinski-Harabasz index (CHI), and Davies-Bouldin index (DBI).
    # The comparison is performed on data prepared by a ClusteringExperiment (a part of the PyCaret library),
    # and the function determines the optimal number of clusters for each clustering model for every configuration.
    #     In summary, this function is used in the compare_normalization_and_transformation_setups function to dynamically
    #     determine the optimal number of clusters for each configuration of normalization and transformation methods being compared.

    # IMPORTANT: Uncomment the follow in the final run
    # norm_tran_res = compare_normalization_and_transformation_setups(normalization_and_transformation_setups,
    #                                                                 trace_dataframe,
    #                                                                 schema['index'],
    #                                                                 schema['categorical_features'],
    #                                                                 schema['ordinal_features'],
    #                                                                 schema['numeric_features'],
    #                                                                 schema['datetime_index_columns'])

    # The function compare_normalization_transformation_pca_setups is similar to the compare_normalization_and_transformation_setups
    # function but with an additional parameter to evaluate:
    # the number of PCA components. Principal Component Analysis (PCA) is a technique often used for dimensionality reduction before
    # performing machine learning tasks.
    norm_tran_pca_res = compare_normalization_transformation_pca_setups(pca_components_setups,
                                                                        trace_dataframe,
                                                                        schema['index'],
                                                                        schema['categorical_features'],
                                                                        schema['ordinal_features'],
                                                                        schema['numeric_features'],
                                                                        schema['datetime_index_columns'])

    # The function calculate_combine_score_and_selecte_models_based_on_median is designed to combine multiple clustering metrics
    # and select the model configurations that are above the median of the combined score.
    selected_models_list = calculate_combine_score_and_select_models_based_on_median(pd.DataFrame(norm_tran_pca_res))

    # The function assigned_clustering is designed to assign clustering labels to the data using the selected model configurations.
    #
    # This process produce trace data based on selected models.
    # The models are chosen based on combined scores ranged from 0 to 1, a higher score is the optimal vale.
    trace_dataframe_clustered_using_selected_models, selected_models = assigned_clustering(trace_dataframe,
                                                      schema['index'],
                                                      schema['categorical_features'],
                                                      schema['ordinal_features'],
                                                      schema['numeric_features'],
                                                      schema['datetime_index_columns'],
                                                                                           selected_models_list)

    # Ensemble clustering of selected model configurations pipelines using their cluster labels.
    # to calculate the SC, we can use PCA and then calculate SC based on each clustering schema.
    labels = trace_dataframe_clustered_using_selected_models[selected_models].to_numpy()

    import category_encoders as ce

    # 'ap' - Affinity Propagation
    # 'meanshift' - Mean shift Clustering
    # 'dbscan' - Density-Based Spatial Clustering
    # 'optics' - OPTICS Clustering

    # 'kmeans' - K-Means Clustering
    # 'hclust' - Agglomerative Clustering
    # 'sc' - Spectral Clustering
    # 'birch' - Birch Clustering
    # 'kmodes' - K-Modes Clusterin
    # meta_model_indexes = ['kmeans', 'hclust', 'sc', 'birch', 'kmodes', 'meanshift', 'ap', 'dbscan', 'optics']
    trace_dataframe_ensemble_clustered_f = trace_dataframe_clustered_using_selected_models
    meta_model_indexes = ['kmeans', 'hclust', 'meanshift']
    Ensemble_Cluster_cols = []
    res_list = []
    for meta_model_index in meta_model_indexes:
            for pca in [False, True]:
                meta_clustering_model, ensemble_labels, res = ensemble_clustering(Cluster_labels=labels,
                                                                             meta_clustering_model_index=meta_model_index,
                                                                             pca=pca,
                                                                             setup_kwargs=dict(
                                                                                 encoding_method=ce.CountEncoder(normalize=False),
                                                                                 max_encoding_ohe=1,
                                                                                 experiment_name=f"Ensemble Clustering",
                                                                                 log_experiment=True,
                                                                                 # log_plots=['elbow', 'silhouette','distance'],
                                                                                 html=False,
                                                                                 memory='./caching_directory'),
                                                                             meta_model_kwargs={},
                                                                             session_id=session_id)
                trace_dataframe_ensemble_clustered = trace_dataframe_clustered_using_selected_models.merge(
                    ensemble_labels,
                    left_index=True,
                    right_index=True)
                trace_dataframe_ensemble_clustered_f = trace_dataframe_ensemble_clustered_f.merge(
                    ensemble_labels,
                    left_index=True,
                    right_index=True)
                Ensemble_Cluster_cols.append(ensemble_labels.name)
                res_list.append(res)
                trace_dataframe_ensemble_clustered.to_csv(
                    f'./out/trace_dataframe_ensemble_clustered_{meta_model_index}_{pca}.csv')
    # ----------------------------------
    # The result of ensemble learning outcomes in the respect of base clustering outcomes in the respect of base clustering outcomes
    res_df = pd.DataFrame(res_list)
    res_df_ = res_df[['meta_clustering_model_name', 'num_clusters', 'pca', 'SC', 'CHI', 'DBI']]
    res_df_.columns = ['Meta Model', 'Clusters', 'PCA', 'SC', 'CHI', 'DBI']
    res_df_.reset_index(inplace=True, drop=True)
    res_df_.to_latex(buf=f'out_tbl/meta_clustering_score_results.tex',
                     caption=f'Comparison result of ensemble clustering in the respect of the base clustering outcomes',
                     label=f'tbl:meta_clustering_score_results', position='t',
                     index=False, escape=False)

    # ----------------------------------
    res_df_normalized = res_df_.copy()
    import joblib
    silhouette_scaler = joblib.load('./scores/silhouette_scaler.pkl')
    calinski_harabasz_scaler = joblib.load('./scores/calinski_harabasz_scaler.pkl')
    davies_bouldin_scaler = joblib.load('./scores/davies_bouldin_scaler.pkl')

    res_df_normalized['normalized_SC'] = silhouette_scaler.transform(res_df_normalized[['SC']])
    res_df_normalized['normalized_CHI'] = calinski_harabasz_scaler.transform(res_df_normalized[['CHI']])
    res_df_normalized['normalized_DBI'] = 1 - davies_bouldin_scaler.transform(res_df_normalized[['DBI']])

    weight_silhouette = 0.34
    weight_calinski_harabasz = 0.33
    weight_davies_bouldin = 0.33

    # Combine the scores using weighted average
    res_df_normalized['combined_score'] = (weight_silhouette * res_df_normalized['normalized_SC'] +
                                   weight_calinski_harabasz * res_df_normalized['normalized_CHI'] +
                                   weight_davies_bouldin * res_df_normalized['normalized_DBI'])

    res_df_normalized['normalized_SC'] = res_df_normalized['normalized_SC'].mul(100).round(2).astype(str).add(' \%')
    res_df_normalized['normalized_CHI'] = res_df_normalized['normalized_CHI'].mul(100).round(2).astype(str).add(' \%')
    res_df_normalized['normalized_DBI'] = res_df_normalized['normalized_DBI'].mul(100).round(2).astype(str).add(' \%')
    res_df_normalized['combined_score'] = res_df_normalized['combined_score'].mul(100).round(2).astype(str).add(' \%')

    res_df_normalized.to_latex(buf=f'out_tbl/meta_clustering_score_results_normalized.tex',
                     caption=f'Comparison result of ensemble clustering in the respect of the base clustering outcomes (Normalized)',
                     label=f'tbl:meta_clustering_score_results_normalized', position='t',
                     index=False, escape=False)
    # -----------------------------------

    # Final evaluation
    from category_encoders import CountEncoder

    # from pycaret.clustering import ClusteringExperiment

    SP = ClusteringExperiment()
    SP.setup(data=trace_dataframe, index=schema['index'],
             categorical_features=schema['categorical_features'],
             encoding_method=CountEncoder(), max_encoding_ohe=3,
             ordinal_features=schema['ordinal_features'],
             numeric_features=schema['numeric_features'],

             date_features=schema['datetime_index_columns'],
             create_date_columns=['weekday', 'day', 'hour', 'minute'],

             # Remove zero variance and perfect remove col-linearity
             low_variance_threshold=0,  # keep all features with non-zero variance,
             remove_multicollinearity=True,
             # Minimum absolute Pearson correlation to identify correlated features. The default value removes equal columns.
             multicollinearity_threshold=0.99,

             transformation=False,

             normalize=False,
             pca=True,
             pca_components=1,

             session_id=session_id,
             experiment_name=f"SP Dataset",
             log_experiment=True,
             log_plots=False,
             html=False,
             memory='./caching_directory'
             )

    import sklearn.metrics as metrics
    from sklearn.preprocessing import LabelEncoder
    # from yellowbrick.text import UMAPVisualizer
    # from yellowbrick.text import TSNEVisualizer

    res_respect_original_list = []
    for l in Ensemble_Cluster_cols:
        res_respect_original = dict()
        encoded_cluster_vector = LabelEncoder().fit_transform(trace_dataframe_ensemble_clustered_f[l])
        res_respect_original["meta_model_index"] = l.split('_')[-2]
        res_respect_original["PCA"] = l.split('_')[-1]
        res_respect_original['SC'] = metrics.silhouette_score(SP.X_transformed, encoded_cluster_vector)
        res_respect_original['CHI'] = metrics.calinski_harabasz_score(SP.X_transformed, encoded_cluster_vector)
        res_respect_original['DBI'] = metrics.davies_bouldin_score(SP.X_transformed, encoded_cluster_vector)
        res_respect_original_list.append(res_respect_original)


    res_df_ens = pd.DataFrame(res_respect_original_list)
    # Comparison result of ensemble clustering in the respect of the original data (assigned clustering outcome to the original data)
    res_df_ens.columns = ['Meta Model', 'PCA', 'SC', 'CHI', 'DBI']
    res_df_ens.to_latex(buf=f'out_tbl/meta_clustering_score_results_respect_original.tex',
                     caption=f'Comparison result of ensemble clustering in the respect of the original data ',
                     label=f'tbl:meta_clustering_score_results_respect_original', position='t',
                     index=False, escape=False)

    print(res_df_ens)

    # ----------------------------------
    res_df_ens_normalized = res_df_ens.copy()
    import joblib
    silhouette_scaler = joblib.load('./scores/silhouette_scaler.pkl')
    calinski_harabasz_scaler = joblib.load('./scores/calinski_harabasz_scaler.pkl')
    davies_bouldin_scaler = joblib.load('./scores/davies_bouldin_scaler.pkl')

    res_df_ens_normalized['normalized_SC'] = silhouette_scaler.transform(res_df_ens_normalized[['SC']])
    res_df_ens_normalized['normalized_CHI'] = calinski_harabasz_scaler.transform(res_df_ens_normalized[['CHI']])
    res_df_ens_normalized['normalized_DBI'] = 1 - davies_bouldin_scaler.transform(res_df_ens_normalized[['DBI']])

    weight_silhouette = 0.34
    weight_calinski_harabasz = 0.33
    weight_davies_bouldin = 0.33

    # Combine the scores using weighted average
    res_df_ens_normalized['combined_score'] = (weight_silhouette * res_df_ens_normalized['normalized_SC'] +
                                   weight_calinski_harabasz * res_df_ens_normalized['normalized_CHI'] +
                                   weight_davies_bouldin * res_df_ens_normalized['normalized_DBI'])

    res_df_ens_normalized['normalized_SC'] = res_df_ens_normalized['normalized_SC'].mul(100).round(2).astype(str).add(' \%')
    res_df_ens_normalized['normalized_CHI'] = res_df_ens_normalized['normalized_CHI'].mul(100).round(2).astype(str).add(' \%')
    res_df_ens_normalized['normalized_DBI'] = res_df_ens_normalized['normalized_DBI'].mul(100).round(2).astype(str).add(' \%')
    res_df_ens_normalized['combined_score'] = res_df_ens_normalized['combined_score'].mul(100).round(2).astype(str).add(' \%')

    res_df_ens_normalized.to_latex(buf=f'out_tbl/meta_clustering_score_results_respect_original_normalized.tex',
                     caption=f'Comparison result of ensemble clustering in the respect of the original data (Normalized)',
                     label=f'tbl:meta_clustering_score_results_respect_original_normalized', position='t',
                     index=False, escape=False)
    # -----------------------------------

    SP = ClusteringExperiment()
    SP.setup(data=trace_dataframe, index=schema['index'],
             categorical_features=schema['categorical_features'],
             encoding_method=CountEncoder(), max_encoding_ohe=3,
             ordinal_features=schema['ordinal_features'],
             numeric_features=schema['numeric_features'],

             date_features=schema['datetime_index_columns'],
             create_date_columns=['weekday', 'day', 'hour', 'minute'],

             # Remove zero variance and perfect remove col-linearity
             low_variance_threshold=0,  # keep all features with non-zero variance,
             remove_multicollinearity=True,
             # Minimum absolute Pearson correlation to identify correlated features. The default value removes equal columns.
             multicollinearity_threshold=0.99,

             transformation=False,

             normalize=False,
             # pca=True,
             # pca_components=1,

             session_id=session_id,
             experiment_name=f"SP - Baseline",
             log_experiment=True,
             log_plots=False,
             html=False,
             memory='./caching_directory'
             )
    SP.remove_metric('hs')
    SP.remove_metric('ari')
    SP.remove_metric('cs')
    
    Baseline_res = []
    
    clustering_model_index = 'kmeans'
    clustering_model_name = 'KMeans'
    baseline_kmeans = SP.create_model(clustering_model_index)
    res = dict(clustering_model_index=clustering_model_index,
               clustering_model_name=clustering_model_name,
               num_clusters=np.unique(baseline_kmeans.labels_).size,
               SC=SP.pull()['Silhouette'][0],
               CHI=SP.pull()['Calinski-Harabasz'][0],
               DBI=SP.pull()['Davies-Bouldin'][0])
    Baseline_res.append(res)

    clustering_model_index = 'meanshift'
    clustering_model_name = 'MeanShift'
    baseline_meanshift = SP.create_model(clustering_model_index)
    res = dict(clustering_model_index=clustering_model_index,
               clustering_model_name=clustering_model_name,
               num_clusters=np.unique(baseline_meanshift.labels_).size,
               SC=SP.pull()['Silhouette'][0],
               CHI=SP.pull()['Calinski-Harabasz'][0],
               DBI=SP.pull()['Davies-Bouldin'][0])
    Baseline_res.append(res)

    clustering_model_index = 'hclust'
    clustering_model_name = 'Agglomerative'
    baseline_hclust = SP.create_model(clustering_model_index)
    res = dict(clustering_model_index=clustering_model_index,
               clustering_model_name=clustering_model_name,
               num_clusters=np.unique(baseline_hclust.labels_).size,
               SC=SP.pull()['Silhouette'][0],
               CHI=SP.pull()['Calinski-Harabasz'][0],
               DBI=SP.pull()['Davies-Bouldin'][0])
    Baseline_res.append(res)

    clustering_model_index = 'optics'
    clustering_model_name = 'OPTICS'
    baseline_optics = SP.create_model(clustering_model_index)
    res = dict(clustering_model_index=clustering_model_index,
               clustering_model_name=clustering_model_name,
               num_clusters=np.unique(baseline_optics.labels_).size,
               SC=SP.pull()['Silhouette'][0],
               CHI=SP.pull()['Calinski-Harabasz'][0],
               DBI=SP.pull()['Davies-Bouldin'][0])
    Baseline_res.append(res)

    clustering_model_index = 'birch'
    clustering_model_name = 'Birch'
    baseline_birch = SP.create_model(clustering_model_index)
    res = dict(clustering_model_index=clustering_model_index,
               clustering_model_name=clustering_model_name,
               num_clusters=np.unique(baseline_birch.labels_).size,
               SC=SP.pull()['Silhouette'][0],
               CHI=SP.pull()['Calinski-Harabasz'][0],
               DBI=SP.pull()['Davies-Bouldin'][0])
    Baseline_res.append(res)

    clustering_model_index = 'kmodes'
    clustering_model_name = 'KModes'
    baseline_kmodes = SP.create_model(clustering_model_index)
    res = dict(clustering_model_index=clustering_model_index,
               clustering_model_name=clustering_model_name,
               num_clusters=np.unique(baseline_kmodes.labels_).size,
               SC=SP.pull()['Silhouette'][0],
               CHI=SP.pull()['Calinski-Harabasz'][0],
               DBI=SP.pull()['Davies-Bouldin'][0])
    Baseline_res.append(res)

    Baseline_res_df =pd.DataFrame(Baseline_res)
    Baseline_res_df.columns = ['index','Model', 'Clusters', 'SC', 'CHI', 'DBI']
    Baseline_res_df.drop('index', axis=1, inplace=True)
    Baseline_res_df.to_latex(buf=f'out_tbl/Baseline_res.tex',
                                   caption=f'Comparison result of baseline clustering algorithms with respect to the transformed original data using the standard pipeline',
                                   label=f'tbl:Baseline_res', position='t',
                                   index=False, escape=False)

    print("Done")


# Using the special variable
# __name__
if __name__ == "__main__":
    main()
