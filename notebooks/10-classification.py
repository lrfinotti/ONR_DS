# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Classification

# %% [markdown]
# ## Acknowledgment 
#
# Some of the content here is based on [Computational and Inferential Thinking: The Foundations of Data Science](https://inferentialthinking.com/chapters/intro.html), by A. Adhikari, J. DeNero, D. Wagner.
#
# On the other hand, this text uses its own module `datascience` for data frame manipulations, while we will use pandas, which is the most commonly used library for data frames in Python.

# %% [markdown]
# ## Importing Modules
#
# Let's start by importing the necessary modules:

# %%
import numpy as np

import pandas as pd
# pd.options.future.infer_string = True
# pd.options.mode.copy_on_write = True

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 6)  # default figure size
plt.style.use("ggplot")  # style sheet

from tqdm.notebook import tqdm

# %% [markdown]
# ## Introduction
#
# Sometimes we want to use data to make predictions.  Here we will discuss a method of prediction for two possible outcomes, instead of a numerical value.  
#
# This sort of prediction is referred to as *classification*.
#
# Here are some examples:
#
# * Does a person have a particular disease or not?
# * Is a bank note a counterfeit or not?
# * Does a picture show a cat or not?
#
# In all these examples you want to use compare a new piece of data with previous instances, for which you know the answer, to decide on the classification of the new data.
#
# For instance, in the first example, you can use data about exams or imaging from already (correctly) diagnosed patients, and compare the data from the new patient to this new data to make the decision.
#
# Here we will deal only with *binary classification*, i.e., having only two classes for our classification.
#
# In a classification task, each individual or situation where we'd like to make a prediction is called an *observation*.  We ordinarily have many observations.  
#
# Each observation has multiple *attributes*, which are known (for example, white blood cells count or weight for a patient).  
#
# Also, each observation has a *class*, which is the answer to the question we care about (for example, fraudulent or not, or picture of a cat or not).
#

# %% [markdown]
# ## Nearest Neighbor Classifier
#
# Let's look at an example.  The file [banknote.csv](banknote) (provided with this notebook) contains data collected by researchers, based on photographs of many individual banknotes: some counterfeit, some legitimate.  They computed a few numbers from each image, using techniques that we won't worry about here.  So, for each banknote, we know a few numbers that were computed from a photograph of it as well as its class: counterfeit, marked as Class 1, or not, marked as Class 0.

# %%
banknotes = pd.read_csv("banknote.csv")
banknotes

# %% [markdown]
# Let's add a color column, to help us draw scatter plots where we can distinguish the two classes by color.  We will use gold for counterfeit notes (Class 0) and dark blue for legitimate notes (Class 1):

# %%
color_table = pd.DataFrame(
    {"Class": np.array([1, 0]), "Color": np.array(["darkblue", "gold"])}
)

banknotes = pd.merge(banknotes, color_table, how="left")

banknotes


# %% [markdown]
# Now let's write a function that allows us to pass a color column together with the scatter plot data, to produce colored scatter plots:

# %%
def scatter_group_color(df, x, y, color, **kwargs):
    """
    Uses a particular column with colors to split and color scatter
    graphs.

    INPUTS:
    df: data frame used for the scatter plot;
    x: label for the x-values;
    y: label for the y-values;
    color: label for the color column.

    OUTPUT:
    Scatter plot of df with colors according to color column.
    """
    fig, ax = plt.subplots()

    for c in df[color].unique():
        df_t = df.loc[df[color] == c]
        df_t.plot(x, y, kind="scatter", color=c, ax=ax, **kwargs)


# %% [markdown]
# Let's look at how the attributes `WaveltVar` and `WaveletCurt` relate to the classification:

# %%
scatter_group_color(banknotes, "WaveletVar", "WaveletCurt", "Color")

# %% [markdown]
# As we can see, points to the right tend to be classified as 0 (legitimate), while points to the left tend to be classified as 1 (counterfeit), so you can see how a new bank note falling in either of those areas could be classified.  On the other hand, a new note that would fall in the middle region would be more difficult to classify.
#
# But we are dropping some attributes!  For instance, adding the `WaveletSkew` attribute, we might be able to separate points that were very close before:

# %%
ax = plt.figure(figsize=(8, 8)).add_subplot(111, projection="3d")
ax.scatter(
    banknotes["WaveletVar"],
    banknotes["WaveletCurt"],
    banknotes["WaveletSkew"],
    c=banknotes["Color"],
)


ax.set_xlabel("WaveletVar")
ax.set_ylabel("WaveletCurt")
ax.set_zlabel("WaveletSkew");

# %% [markdown]
# Let's visualize how this with a specific example.  For instance, consider the new note:

# %%
new_note = np.array([0, 0, 0, 0])

# %% [markdown]
# With two attributes, it would be hard to make sure which classification we should give it:

# %%
fig, ax = plt.subplots()

for c in banknotes["Color"].unique():
    df_t =banknotes.loc[banknotes["Color"] == c]
    df_t.plot("WaveletVar", "WaveletCurt", kind="scatter", color=c, ax=ax)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

ax.scatter(new_note[0], new_note[2], c="red");

# %% [markdown]
# But, adding a third attribute (`WaveletSkew`) we can add depth, and points that were basically on top of each other get separated:

# %%
ax = plt.figure(figsize=(8, 8)).add_subplot(111, projection="3d")
ax.scatter(
    banknotes["WaveletVar"],
    banknotes["WaveletCurt"],
    banknotes["WaveletSkew"],
    c=banknotes["Color"],
    zorder=1
)

ax.scatter(new_note[0], new_note[2], new_note[1], c="red", zorder=5)

ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(-4, 4)

ax.set_xlabel("WaveletVar")
ax.set_ylabel("WaveletCurt")
ax.set_zlabel("WaveletSkew");

# %% [markdown]
# It is now clear that we should classify this note as *legitimate* (dark blue color.)
#
# So, it is best to use as many attributes as we have!
#
#
# ### Classification
#
# An initial idea for our classification then, is to just look at the *closest* point to our new one, and classify with the same class as this closest point.
#
# Before we can write the code for these computations, we need to discuss how to compute distances.

# %% [markdown]
# ## Measuring Distance
#
# To measure the distance between two data points with two attributes (as in our example above), say $(x_0, y_0)$ and $(x_1, y_1)$, we simply compute
#
# $$
# \sqrt{(x_1-x_0)^2 + (y_1-y_0)^2}.
# $$
#
# This is simply the Pythagorean theorem: we have a right triangle with side lengths $|x_1-x_0|$ and $|y_1-y_0|$, and we want to find the length of the hypotenuse:
#
# ![distance](distance.png)
#
# If we had two points in three dimensional space, e.g., if we had three attributes, say $(x_0, y_0, z_0)$ and $(x_1, y_1, z_1)$, then we can apply the Pythagorean theorem *twice* and obtain distance
#
# $$
# \sqrt{(x_1-x_0)^2 + (y_1-y_0)^2 + (z_1 - z_0)^2}.
# $$
#
# Although geometrically we can't quite visualize more dimensions, in practice we might have more than three attributes, and we still need to compute distances.  And we *can* do it! It follows a similar pattern: the distance between $(x_1, x_2, \ldots, x_n)$ and $(y_1, y_2, \ldots, y_n)$ is given by
#
# $$
# \sqrt{(x_1-y_1)^2 + (x_2-y_2)^2 + \cdots +  (x_n - y_n)^2}.
# $$
#
# **Bottom line:** To compute the distance between two points (in any number of dimensions):
#
# * subtract their coordinates;
# * square these differences;
# * add all these squares;
# * take the square root of this sum.

# %% [markdown]
# ## Example
#
# Let's now find the nearest point to a new given one.  We have four attributes, in this case.  Let's say our new data point is:

# %%
new_note = np.array([2, 5, -4, 0])

# %% [markdown]
# Let's compute distances from every bank note in our data frame to this new note:

# %%
# convert attributes to an array
attributes_array = banknotes.iloc[:, 0:4].to_numpy()

# compute the distance
distances = np.sqrt(
    np.sum(
        (attributes_array - new_note) ** 2,
        axis=1  # add rows!
    )
)

distances

# %% [markdown]
# Let's add distances to our data frame:

# %%
banknotes_with_dist = banknotes.assign(Distance=distances)

banknotes_with_dist

# %% [markdown]
# To find the closest one, we sort by distance and take the first one:

# %%
banknotes_with_dist.sort_values("Distance")

# %% [markdown]
# As we can see, the closest point is classified as 1 (legitimate), so we would classify this new note as legitimate as well.
#
# Also observe that we had some counterfeit notes not too far away!
#
# This process of classification can also be automated, which we do below.

# %% [markdown]
# ## $k$ Nearest Neighbors
#
# If we get a new point that is closely surrounded by points of both classifications, simply choosing the classification of the nearest point is prone to incorrect classification.  Although this cannot be entirely avoid in a situation like this, an idea that could improve the results is to look at *a few* of the closest points, and classify the new one according to the *majority*.
#
# For instance, in our previous example:

# %%
banknotes_with_dist.sort_values("Distance").head(10)


# %% [markdown]
# Of the ten closest points to our new note, seven were legitimate notes (class 1), so we would classify it as legitimate as well.  Although in this case looking at more points does not change our classification, one can certainly see how it could happen.
#
# **Note:** Usually we take an odd number of points, so that are no ties when we take the classification of the majority.
#
# Let's now work on the implementation.

# %% [markdown]
# ## Implementation
#
# ### Distances
#
# As in our concrete example above, the first step is to create a data frame with all distances, sorted.

# %%
def df_with_dists(df, attributes, new_point):
    """
    Given a data frame and a list of the attributes labels, adds a column with 
    distances between the rows of df and the new_point and sort it by these
    distances.

    INPUTS:
    df: data frame to add distances and sort;
    attributes: list with labels for attributes;
    new_point: point to which we compute distances.

    OUTPUT:
    df with column of distances to new_point, sorted by distance.
    """
    # convert attributes to an array
    attributes_array = df[attributes].to_numpy()

    # compute the distance
    distances = np.sqrt(
        np.sum(
            (attributes_array - new_point) ** 2,
            axis=1  # add rows!
        )
    )

    return df.assign(Distance=distances).sort_values("Distance")


# %% [markdown]
# Let's test it with the previous example:

# %%
note_attr = ["WaveletVar", "WaveletSkew", "WaveletCurt", "Entropy"]
df_with_dists(banknotes, note_attr, new_note)


# %% [markdown]
# ### Majority Vote
#
# Let's now write a function that takes a *series* (column of the data frame) of classifications, and returns the majority.
#
# If we are dealing with only 0 and 1 in the classification, we can simply round the average the classifications to the nearest integer.  But let's make the code more robust here: it will really return the most common value.  (The classifications can be any objects, and there can even be more than one class.)

# %%
def majority_class(classifications, k):
    """
    Returns the most frequent value of the first k entries of classifications.

    INPUTS:
    classifications: series containing the classifications;
    k: number of nearest neighbors to use in majority classification.

    OUTPUT:
    Most occurred class among the first k in classifications.
    """
    return classifications.head(k).value_counts().index[0]


# %% [markdown]
# Now, we can combine the last two to make a single function that does the whole job:

# %%
def k_nearest_neighbors(df, attributes, new_point, k, class_label="Class"):
    """
    Classify new_point using the k nearest points.

    INPUTS:
    df: data frame used for classification;
    attributes: list of labels for the attribute columns;
    new_point: new point to be classified;
    k: number of neared neighbors;
    class_label: label for classification column.

    OUTPUT:
    Classification of new_point using k nearest neighbors.
    """
    classifications = df_with_dists(df, attributes, new_point)[class_label]
    return majority_class(classifications, k)


# %% [markdown]
# Using the same new note:

# %%
k_nearest_neighbors(banknotes, note_attr, new_note, 5)


# %% [markdown]
# ## Testing the Classifier
#
# Now, we need to test our classifier.  With this method, it does not help to reclassify the points we used in our classifier, since each point will be nearest to itself.  The standard practice in this situation is to split the data frame into a two *disjoint* subsets:
#
# * *training set*: a subset for which we use the $k$ nearest neighbors method;
# * *testing set*: a set we classify with our classifier, and verify the accuracy.
#
# These two sets should be split randomly, with some given proportions.
#
# Let's write a function for the splitting:

# %%
def split_df(df, first_prop):
    """
    Splits df randomly in two: one with proportion first_prop, and
    the other containing the rest.

    INPUTS:
    df: data frame to split;
    firt_prop: proportion of the original data frame to use for the first output.

    OUTPUT:
    Two random subsets of df, with first being of first_prop proportion.
    """

    df1 = df.sample(frac=first_prop)

    df2 = df.drop(df1.index)

    return df1.reset_index(drop=True), df2.reset_index(drop=True)


# %% [markdown]
# Let's split in 60/40 between training and testing:

# %%
training_df, testing_df = split_df(banknotes, 0.6)

# %%
len(training_df)

# %%
len(testing_df)

# %% [markdown]
# Now, we classify each entry of the testing data frame.  Let's write a function to help:

# %%
k = 5

# classify all rows
classifications = testing_df[note_attr].apply(
    lambda row: k_nearest_neighbors(training_df, note_attr, row.to_numpy(), k), axis=1
)

# add classifications as new column
testing_df_results = testing_df.assign(Result=classifications)

testing_df_results

# %% [markdown]
# Let's see the proportion of *correctly* classified notes:

# %%
np.count_nonzero(testing_df_results["Result"] == testing_df_results["Class"]) / len(
    testing_df_results
)


# %% [markdown]
# We obtained 100% accuracy!

# %% [markdown]
# Let's put this testing into a function as well:

# %%
def test_results_accuracy(training_df, testing_df, attributes, k):
    """
    Given training and testing data frames, classifiy the testing data frame using
    k nearest neighbors and returns the proportion of correct classifications.

    INPUTS:
    training_df: training data frame;
    testing_df: testing data frame;
    attributes: list of attribute column labels;
    k: number of nearest neighbors.

    OUTPUT:
    Proportion of correctly classified rows of testing_df.
    """
    classifications = testing_df[attributes].apply(
        lambda row: k_nearest_neighbors(training_df, attributes, row.to_numpy(), k),
        axis=1,
    )

    testing_df_results = testing_df.assign(Result=classifications)

    return np.count_nonzero(
        testing_df_results["Result"] == testing_df_results["Class"]
    ) / len(testing_df_results)


# %%
test_results_accuracy(training_df, testing_df, note_attr, 5)

# %% [markdown]
# ### Speed vs Accuracy
#
# As the example illustrates, this $k$ nearest neighbors method can be extremely accurate!
#
# On the other hand, the classification in (relatively) quite slow.  For every new data point, we need to calculate its distance to the whole training data frame (which can be large), sort, and select the majority.
#
# With modern computers (and not so large training sets), the computation time can be more than tolerable, though.

# %% [markdown]
# ## Example: Detecting Brest Cancer
#
# This example (from [Computational and Inferential Thinking: The Foundations of Data Science](https://inferentialthinking.com/chapters/intro.html), by A. Adhikari, J. DeNero, D. Wagner) is based on Brittany Wenger's [science fair project](https://sites.google.com/a/googlesciencefair.com/science-fair-2012-project-64a91af142a459cfb486ed5cb05f803b2eb41354-1333130785-87/home) of building a classification algorithm to diagnose breast cancer.  She won grand prize for building an algorithm whose accuracy was almost 99%.
#
# Here we will use $k$ nearest neighbors.
#
# Exams give us data to try to classify tumors as benign (class 0) or malignant (class 1).  The data is in the file [breast-cancer.csv](breas-cancer.csv) provided with this notebook.

# %%
patients = pd.read_csv("breast-cancer.csv").drop(columns="ID")
patients

# %% [markdown]
# First, we split the data frame into training and testing.  Let's use 60/40 again:

# %%
training_df, testing_df = split_df(patients, 0.5)

# %% [markdown]
# Now, let's test how well it did:

# %%
patients_attr = list(patients.columns[:-1])

test_results_accuracy(training_df, testing_df, patients_attr, 5)

# %% [markdown]
# Not bad!

# %% [markdown]
# ## Comments, Suggestions, Corrections
#
# Please send your comments, suggestions, and corrections to lfinotti@utk.edu.
