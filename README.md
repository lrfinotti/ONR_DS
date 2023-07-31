# Data Analysis with Python

We provide a series of [Jupyter Notebooks](https://jupyter.org/) introducing [Python](https://jupyter.org/) tools for [Data Science](https://en.wikipedia.org/wiki/Data_science).

The full set of notebooks and necessary data sets can be found in the found in the file [notebooks.zip](notebooks.zip).


### Videos

I've made a few quick videos for each notebook.  A playlist with all the videos can be found here: [Python for Data Science](https://youtube.com/playlist?list=PL0hG_ZfAGizsv8VvBSwgu7zUVhko35cUE).  There are links for each individual video (together with the corresponding notebook) below.

**Note:** Although I believe that the notebooks provide a good introduction to the topics, the accompanying videos are *less than perfect* (to say the least).  Hopefully they can still provide an adequate exposition of the main topics of each notebook.  (I intend to rerecord them with better quality in the future, if time allows.)


### Acknowledgment

Many of the topics and examples in these notebooks were inspired by (or simply taken from) the book [Computational and Inferential Thinking: The Foundations of Data Science](https://inferentialthinking.com/chapters/intro.html), by A. Adhikari, J. DeNero, D. Wagner.

On the other hand, while this text uses its own module [datascience](http://www.data8.org/datascience/) for data frame manipulations, [pandas](https://pandas.pydata.org/) is used in the notebooks provided.


## Topics

### 1 - Introduction to Python and Jupyter Notebooks for Data Science

- Notebook: [01-python.ipynb](notebooks/01-python.ipynb)
- Video: https://youtu.be/ZqEXBWzI9jA (*Length:* 3:03:44.)

This notebook provides an brief introduction to Jupyter Lab and basic Python aimed towards applications in Data Science.

Topics include:

1. Jupyter Notebooks
2. Numbers and Computations
3. Strings
4. Lists
5. Dictionaries
6. Conditionals
7. for Loops
8. Functions


### 2 - NumPy

- Notebook: [02-numpy.ipynb](notebooks/02-numpy.ipynb)
- Video: https://youtu.be/wf5Ct5UOZFU (*Length:* 1:31:47)

This notebooks introduces computations and manipulation of [NumPy](https://numpy.org/) arrays.

Topics include:

1. Creating Arrays
2. Operations with Arrays and NumPy functions
3. Length, Slicing, Filtering, Counting, Equality.
4. Arrays of Strings
5. Efficiency
6. Examples:
   1. Converting Temperatures
   2. Checking a Trigonometric Identity
   3. Leibniz Formula
   4. Grade Computation
   5. Compound Interest

### 3 - Data Frames with Pandas

- Notebook: [03-data_frames.ipynb](notebooks/03-data_frames.ipynb)
- Data Sets:
  - [nba_salaries.csv](notebooks/nba_salaries.csv)
- Video: https://youtu.be/AZpzCXxbn4Q (*Length:* 1:29:11)

This notebook introduces the [pandas](https://pandas.pydata.org/) library for manipulation of data frames.

Topics include:

1. Creating and Reading Data Frames
2. Modifying Data Frames
3. Selecting Rows and Columns
4. Filtering
5. Adding Columns
6. Grouping by Categories
7. Pivot Tables


### 4 - Data Visualization

- Notebook: [04-visualization.ipynb](notebooks/04-visualization.ipynb)
- Data Sets:
  - [actors.csv](notebooks/actors.csv)
  - [top_movies_2017.csv](notebooks/top_movies_2017.csv)
  - [usa_ca_2019.csv](notebooks/usa_ca_2019.csv)
- Video: https://youtu.be/oBZlmyi_d54 (*Length:* 1:46:35)

This notebook introduces the use of [MatPlotLib](https://matplotlib.org/) and pandas to visualize data.

Topics Include:

1. Line Plots
2. Scatter Plots
3. Bar Charts
4. Histograms
5. Overlaying Plots


### 5 - Randomness and Probabilities

- Notebook: [05-randomness.ipynb](notebooks/05-randomness.ipynb)
- Video: https://youtu.be/7mdJp3in0mc (1:12:50)

This notebook illustrate the use of Python to simulate random events and compute empirical probabilities.

Topics Include:

1. Randomness with NumPy
2. Tracking and Visualizing Number of Occurrences
3. Values Distributions
4. Computing Empirical Probabilities
5. Examples:
   1. Tossing coins
   2. Rolling dice
   3. Poker


### 6 - Testing Hypotheses

- Notebook: [06-hypothesis.ipynb](notebooks/06-hypothesis.ipynb)
- Video: https://youtu.be/g8ZoaQASeec (*Length:* 46:09)

This notebooks shows how to use Python tools to check if a hypothesis is consistent with a proposed model.

Topics Include:

1. Likelihoods from Probabilities
2. Null Hypothesis and *p*-Value
3. Examples:
   1. Die Biased Towards Sixes
   2. Other Biases
   3. Robert Swain's Case



### 7 - A/B Testing

- Notebook: [07-ab_testing.ipynb](notebooks/07-ab_testing.ipynb)
- Data Sets:
  - [baby.csv](notebooks/baby.csv)
  - [bta.csv](notebooks/bta.csv)
- Video: https://youtu.be/ibHyLOfdlVc (*Length:* 35:33)

This notebook shows how to use Python to see if the values of a binary categorical attribute has an effect on another attribute.

Topics include:

1. Randomizing Labels
2. Examples:
   1. Smoking and Birth Weight
   2. Drug Trial (Chronic Back Pain)


### 8 - Inference

- Notebook: [08-inference.ipynb](notebooks/08-inference.ipynb)
- Data Sets:
  - [united_summer2015.csv](notebooks/united_summer2015.csv)
  - [baby.csv](notebooks/baby.csv)
  - [hodgkins.csv](notebooks/hodgkins.csv)
- Video: https://youtu.be/YYCAkQzJ4kQ (*Length:* 36:51)

This notebooks shows how a sample can give good approximations of certain statistics and how we can estimate the variation of a statistic from the randomness of sample.

Topics include:

1. Sampling
2. Bootstrap
3. Confidence Interval
4. Testing Hypothesis Using Confidence Interval
5. Examples:
   1. Flight Delay
   2. Median Birth Weight


### 9 - Linear Correlation and Predictions

- Notebook: [09-predictions.ipynb](notebooks/09-predictions.ipynb)
- Data Sets:
  - [baby.csv](notebooks/baby.csv)
- Video: https://youtu.be/wNqBHMFkXgc  (*Length:* 1:11:07)

Topics include:

1. Linear Correlation
2. Correlation Coefficient
3. Regression Line
4. Predictions with the Regression Line
5. Root Mean Squared Error
6. Minimizing the RMSE
7. Multiple Attributes
8. Example: Predicting Birth Weight


### 10 - Classification with Nearest Neighbors

- Notebook: [10-classification.ipynb](notebooks/10-classification.ipynb)
- Data Sets:
  - [banknote.csv](notebooks/banknote.csv)
  - [breast-cancer.csv](notebooks/breast-cancer.csv)
- Video: https://youtu.be/t8Q-XNdzfQY (*Length:* 40:31)

This notebook shows how to implement and use the *k* Nearest Neighbors classification method.

Topics include:

1. Distances (in Any Number of Dimensions)
2. Splitting in Training and Testing Sets
3. *k* Nearest Neighbors
4. Implementing the *k* Nearest Neighbors Classification
5. Examples:
   1. Detecting Counterfeit Bank Notes
   2. Diagnosing Breast Cancer
