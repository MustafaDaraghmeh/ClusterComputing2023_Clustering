# Import the required libraries
import os
from driver import load_trace_data
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the default plotting settings
sns.set_theme(context="paper", style='whitegrid',  palette='deep', font='serif', font_scale=1.7, rc={"figure.dpi": 300})

# Random State
session_id = 1000

# number of random samples of VM trace collected from each VM type
number_of_random_samples=5000

try:
    os.mkdir('index_encoders')
except OSError as error:
    print(error)

try:
    os.mkdir('buckets_plots')
except OSError as error:
    print(error)
#
#     The load_trace_data function is designed to load, preprocess, and return a dataset from the Azure VM workload trace.
#     The dataset is loaded from a CSV file and a random sample of rows can be returned using the n parameter.
#     For reproducibility, a random seed (session_id) can be provided.
trace_dataframe_15000, schema_15000 = load_trace_data(n=number_of_random_samples, session_id=session_id)

# Export the dataset
trace_dataframe_15000.to_csv('./trace_dataframe_15000.csv', index=False)

# Plot Cat Plot of original data
print("Plot Cat Plot of sample dataset")
sns.catplot(
    data=trace_dataframe_15000, x="Memory Bucket", y="Life Time (Hour)",
    col="VM Type", aspect=.7, kind="boxen",
    order=['2', '4', '8', '32', '64', '>64'],
    col_order=['Interactive', 'Delay-insensitive', 'Unknown']
)
plt.savefig('buckets_plots/Memory_Bucket_plot_sample.png', dpi=300)
# plt.show()
plt.clf()
plt.cla()
plt.close()

sns.catplot(
    data=trace_dataframe_15000, x="Core Bucket", y="Life Time (Hour)",
    col="VM Type", aspect=.7, kind="boxen",
    order=['2', '4', '8', '24', '>24'],
    col_order=['Interactive', 'Delay-insensitive', 'Unknown']
)
plt.savefig('buckets_plots/Core_Bucket_plot_sample.png', dpi=300)
# plt.show()
plt.clf()
plt.cla()
plt.close()
# --------------------------------------------------

trace_dataframe_None, schema_None = load_trace_data(n=None, session_id=session_id)

# Plot Cat Plot of original data
print("Plot Cat Plot of original dataset")
sns.catplot(
    data=trace_dataframe_None, x="Memory Bucket", y="Life Time (Hour)",
    col="VM Type", aspect=.7, kind="boxen",
    order=['2', '4', '8', '32', '64', '>64'],
    col_order=['Interactive', 'Delay-insensitive', 'Unknown']
)
plt.savefig('buckets_plots/Memory_Bucket_plot.png', dpi=300)
# plt.show()
plt.clf()
plt.cla()
plt.close()

sns.catplot(
    data=trace_dataframe_None, x="Core Bucket", y="Life Time (Hour)",
    col="VM Type", aspect=.7, kind="boxen",
    order=['2', '4', '8', '24', '>24'],
    col_order=['Interactive', 'Delay-insensitive', 'Unknown']
)
plt.savefig('buckets_plots/Core_Bucket_plot.png', dpi=300)
# plt.show()
plt.clf()
plt.cla()
plt.close()
# --------------------------------------------------

print('DONE')