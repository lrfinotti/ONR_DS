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
# # Data Frames with Pandas

# %% [markdown]
# ## Acknowledgment 
#
# Some of the content here is based on [Computational and Inferential Thinking: The Foundations of Data Science](https://inferentialthinking.com/chapters/intro.html), by A. Adhikari, J. DeNero, D. Wagner.
#
# On the other hand, this text uses its own module `datascience` for data frame manipulations, while we will use pandas, which is the most commonly used library for data frames in Python.

# %% [markdown]
# ## Introduction
#
# We call data organized in tables *data frames*.
#
# It can be viewed in two ways:
#
# * a sequence of named columns, each describing a single aspect of all entries in a data set, or
# * a sequence of rows, each containing all information about a single entry in a data set.

# %% [markdown]
# ## Pandas
#
# Our main tool for dealing with data frames will be [pandas](https://pandas.pydata.org/), which is the most popular library for dealing with data frames in Python.
#
# Pandas uses NumPy internally to store the data and perform the operations with rows and columns, and is therefore fast enough to deal with large amounts of data.  (Although there are alternatives better optimized for *huge* amounts of data.)
#
# ### Installing Pandas
#
# pandas does not come with Python, so it needs to be installed separately.  If you have a *vanilla* installation of Python, you can do it by running 
#
# ```
# pip install pandas
# ```
#
# from a terminal.
#
# On the other hand, if you installed Anaconda, it should already be available.
#
# ### Loading Pandas
#
# As usual, we need to first import pandas so that we can use its features.  We usually give the (standard) shortcut `pd`:

# %%
import pandas as pd

# %% [markdown]
# (Thus we can call its functions with `pd.name_of_the_function` instead of `pandas.name_of_the_function`.)

# %% [markdown]
# We also usually need arrays when dealing with  data frames, so let's import NumPy as well:

# %%
import numpy as np

# %% [markdown]
# ## Creating Data Frames
#
# ### Creating Data Frames Manually
#
# We can use *dictionaries* to create a data frames.  The keys are the *labels* for each column (usually a string), and the values are the arrays (or lists) of values for each column.
#
# The syntax is
# ```python
# pd.DataFrame({ 
#     "name of col 1": col_1,
#     "name of col 2": col_2,
#     ...
#     "name of col n": col_n,
#     
#   })
# ```
#
# For example, let's create an array containing information about flowers:

# %%
flowers = pd.DataFrame({
    "Name": np.array(["lotus", "sunflower", "rose"]),
    "Number of petals": np.array([8, 34, 5])

})

flowers

# %% [markdown]
# Note that we automatically get numbers on the left of the rows.  These numbers form the *index* of the data frame, which is used to identify the rows.  As usual, it starts with zero.  Later we will see how we can make some column the index of the data frame.

# %% [markdown]
# ### Reading from Files
#
# Usually we read data from files.  The most common type of file used to store data are [comma-separated values](https://en.wikipedia.org/wiki/Comma-separated_values) files, usually referred to as CSV files.
#
# To get the content of a CSV into a data frame, we use the function `pd.read_csv`, with the name of the file passed as a *string*.  The file has to be in the same folder as our notebook, or the the path to it has to be given in the string.
#
# As an example, let's load the file [nba_salaries.csv](nba_salaries.csv) (provided with this notebook), which contain the [salaries of all National Basketball Association players](https://www.statcrunch.com/app/index.php?dataid=1843341) in 2015-2016, and save the resulting data frame as `nba`:

# %%
nba = pd.read_csv("nba_salaries.csv")
nba

# %% [markdown]
# Each row represents one player. The columns are:
#
# | **Column Label** | **Description**                                      |
# |------------------|------------------------------------------------------|
# | `PLAYER`         | Player's name                                        |
# | `POSITION`       | Player's position on team                            |
# | `TEAM`           | Team name                                            |
# | `'15-'16 SALARY` | Player's salary in 2015-2016, in millions of dollars |
#
# The code for the positions is PG (Point Guard), SG (Shooting Guard), PF (Power Forward), SF (Small Forward), and C (Center).
#
# The first row shows that Paul Millsap, Power Forward for the Atlanta Hawks, had a salary of almost $\$18.7$ million in 2015-2016.

# %% [markdown]
# `pd.read_csv` has many options, which allow you to skip rows at the beginning, choose columns, choose the index, etc.  You can [read the documentation](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) or call
#
# ```python
# help(pd.read_csv)
# ```
#
# Pandas can also read from other formats, including Microsoft Excel (with `pd.read_excel`), but here will only use CSV files.

# %% [markdown]
# ### Reading from URL
#
# We can also use `pd.read_csv` to read a CSV available from a URL.  For instance,
#
# [http://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2019/nc-est2019-agesex-res.csv](http://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2019/nc-est2019-agesex-res.csv)
#
# is a link for a CSV file containing some US Census data.  As long as the link is valid (which is, as of July 2023), passing the URL as a *string* to `pd.read_csv` should load the data frame:

# %%
data_url = "http://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2019/nc-est2019-agesex-res.csv"

census_df = pd.read_csv(data_url)

census_df

# %% [markdown]
# ## Renaming Columns

# %% [markdown]
# Let's relabel the last column simply `SALARY`:

# %%
nba = nba.rename(columns={"'15-'16 SALARY": "SALARY"})  # overwrite nba data frame
nba

# %% [markdown]
# The syntax to rename various columns is:
# ```python
# data_frame_name.rename(
#     columns={
#         old_col_name_1: new_col_name_1,
#         old_col_name_2: new_col_name_2,
#         ...
#         old_col_name_last: new_col_name_last,
#     } )
# ```
#
# In other words, we pass to the method `rename` the argument `columns=` (or it will try to rename the index, by default) with a dictionary whose the key/value pairs contain the old names as keys and the new names as values.

# %% [markdown]
# ## Data Frame Properties

# %% [markdown]
# The attribute `shape` gives a pair: the number of rows and number of columns (in this order):

# %%
nba.shape

# %% [markdown]
# Note that `shape` is an attribute and not a method/function, and therefore *should not* be followed by parentheses.
#
# We can also get the number of rows with `len`, as usual:

# %%
len(nba)

# %% [markdown]
# To get the column names, we can use the attribute `columns`:

# %%
nba.columns

# %% [markdown]
# To get the index of the data frame (which is not very interesting in this case):

# %%
nba.index

# %% [markdown]
# We can also see the data types for each column:

# %%
nba.dtypes

# %% [markdown]
# (Note that `object` is basically a string.)

# %% [markdown]
# We can get the top rows of a data frame using the `head` method.  For instance, to get the top 10 rows:

# %%
nba.head(10)

# %% [markdown]
# We can use `tail` for the bottom rows.  So, to get the bottom 5 rows:

# %%
nba.tail(5)

# %% [markdown]
# ## Changing the Index

# %% [markdown]
# In this case, it might make sense to make the column of players the *index*, as each row has data about a particular player:

# %%
nba_by_player = nba.set_index("PLAYER")

nba_by_player

# %% [markdown]
# The old index is gone and now our rows are identified by the corresponding player's name.

# %% [markdown]
# ## Selecting Columns
#
# We select a column of a data frame by simply passing its label.  For instance, we can select the `SALARY` column of `nba_by_player`:

# %%
nba_by_player["SALARY"]

# %% [markdown]
# Note that the result is *not a data frame*, but a [series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html), which is basically a NumPy array with some extra metadata, such as name (SALARY in this case) and an index.  But, since it is basically a NumPy array, we can use most of NumPy's functions with series.  For instance, if we want to know the average salary in the data frame:

# %%
np.mean(nba_by_player["SALARY"])

# %% [markdown]
# If we need a true NumPy array, we can use `.to_numpy()`:

# %%
nba_by_player["SALARY"].to_numpy()

# %% [markdown]
# On the other hand, the series has its own methods, so we could have found the average salary with:

# %%
nba_by_player["SALARY"].mean()

# %% [markdown]
# Many of the functions provided by NumPy for arrays are available for series as methods, which is the preferred way to use it.

# %% [markdown]
# An alternative is to get columns from the corresponding attributes.  For instance, the following is equivalent to `nba_by_player["SALARY"]`:

# %%
nba_by_player.SALARY

# %% [markdown]
# The problem with this second approach is that the column label cannot have a space.  Due to this limitation, we will always use the first approach (with the brackets `[ ]`).

# %% [markdown]
# We can also select a column by position (*counting from 0*, as usual) using `.iloc`.  For example, to get the second column (TEAM) we can do:

# %%
nba_by_player.iloc[:, 1]

# %% [markdown]
# **Note that `.iloc` is followed by square brackets, not parentheses!**
#
# `.iloc` selects *rows* and columns and we can use *slicing* to select them.  Thus, the `:` in `.iloc[:, 2]` told it to get *all* rows.  (Rows are first, then the columns.)

# %% [markdown]
# We can also select more than one column by passing a list of labels inside the square brackets:

# %%
nba_by_player[["SALARY", "TEAM"]]  # note the double brackets [[ ]]

# %% [markdown]
# Note that we obtain a data frame, the index is preserved, and the order in the given list is used.
#
# In particular, if we want a single row as a data frame, and not a series, we can use the double square brackets again:

# %%
nba[["SALARY"]]  # using nba, not nba_by_player

# %% [markdown]
# We can also use `iloc` to select more than a single columns.  For instance, the last two columns of `nba`:

# %%
nba.iloc[:, -2:]

# %% [markdown]
# Or the first and third, using a list of column indices:

# %%
nba.iloc[:, [0, 2]]

# %% [markdown]
# The index (which is not quite a column) is given by the `.index` attribute (no parentheses!):

# %%
nba_by_player.index

# %% [markdown]
# It's not a series, since it has no index itself, but it is also an array with extra metadata:

# %%
nba_by_player.index.to_numpy()

# %% [markdown]
# ## Selecting Rows
#
# Suppose we want to get the information about a particular players, say Kobe Bryant.
#
# In `nba_by_player`, since the players make the index, we can use `loc`:

# %%
nba_by_player.loc["Kobe Bryant"]

# %% [markdown]
# This gives all the item in Kobe Bryant's row.
#
# **Note:** `loc` filters rows and columns by *labels*, while `iloc` filters by *numerical index*.

# %% [markdown]
# But this would fail with `nba`, in which case the index is numeric, and, a priori, we do not know the number for his row.  (If we did, we could use `iloc`.)  In this case, we can  *filter* for the row.
#
# Similar to NumPy's arrays, we can select rows of a data frame by passing to `loc` an array of booleans.  Thus we can do:

# %%
nba.loc[nba["PLAYER"] == "Kobe Bryant"]

# %% [markdown]
# The result is different (the former was a series, and now we have a data frame), but contains the same information.
#
# In either case, we can get, say, Kobe Bryant's salary:

# %%
nba_by_player.loc["Kobe Bryant"]["SALARY"]  # get SALARY from series

# %%
nba_by_player.loc["Kobe Bryant", "SALARY"]  # index, and column label

# %%
nba.loc[nba["PLAYER"] == "Kobe Bryant"]["SALARY"]

# %% [markdown]
# Note that this last example, the salary comes as a series (since we had a data frame before), so we get the index with the salary as well.  This is the case because the filtering could result in more than one row.

# %% [markdown]
# Again, knowing that the index in `nba` for Kobe Bryant is 169, we could get his row (as a series) with:

# %%
nba.loc[169]

# %% [markdown]
# Note that `iloc` also works, since the index matches the position:

# %%
nba.iloc[169]

# %% [markdown]
# With `iloc` we can get slices of rows:

# %%
nba_by_player.iloc[10:20]  # rows 10 to 19

# %% [markdown]
# And, as seen above, we can also select columns:

# %%
nba_by_player.iloc[10:20, -2:]  # rows 10 to 19, last two columns

# %% [markdown]
# ## Sorting
#
# We can easily sort data frames with the method `sort_values`.  For instance, to sort by SALARY:

# %%
nba_by_player.sort_values("SALARY")

# %% [markdown]
# (Some of the salaries that seem (relatively) quite small are for the player who changed teams mid season, and the data frame only keep their salaries from one team.)
#
# **Note:** The operation above *returns a new data frame*, and does not alter the original.  To do so, you can pass the optional argument `inplace=True`, as in
#
# ```python
# nba_by_player.sort_values("SALARY", inplace=True)
# ```
#
# or simply overwrite the original with
#
# ```python
# nba_by_player = nba_by_player.sort_values("SALARY")
# ```
#
# ***Important:* The same is true for most methods we will introduce here!  They output a new data frame, and do not change the original!**  Some of them will also have the `inplace` optional argument, but you can always just overwrite the original value, as done above, if necessary.

# %% [markdown]
# We can pass the optional argument `ascending=False`, to sort it in decreasing order:

# %%
nba_by_player.sort_values("SALARY", ascending=False)

# %% [markdown]
# We can also sort by the index:

# %%
nba_by_player.sort_index()

# %% [markdown]
# We can sort by more than one column, by passing a list of labels:

# %%
nba_by_player.sort_values(["POSITION", "SALARY"])

# %% [markdown]
# The data is sorted according to the first label, and rows that have the same value for this column are sorted according to the second label.

# %% [markdown]
# ## Resetting the Index
#
# Sometimes when we alter a data frame, a numerical index can get mixed.  For instance, when sorting:

# %%
nba.sort_values("POSITION")

# %% [markdown]
# In these cases often we prefer to *reset the index*:

# %%
nba.sort_values("POSITION").reset_index()

# %% [markdown]
# A new numerical index is given (in the new order), and the old is added a new column, labeled `index` by default.  A name can be given by passing the `names=` argument:

# %%
nba.sort_values("POSITION").reset_index(names="Old Index")

# %% [markdown]
# If we do not want to keep the old index, we can use the optional argument `drop=True`:

# %%
nba.sort_values("POSITION").reset_index(drop=True)

# %% [markdown]
# ## Filtering Rows
#
# ### Filtering with `loc`
#
# As we've seen, we can use `loc` to filter rows by conditions.
#
# For example, to find all rows for centers:

# %%
nba_by_player.loc[nba_by_player["POSITION"] == "C"]

# %% [markdown]
# To filter rows for player who make 20 million or more:

# %%
nba_by_player.loc[nba_by_player["SALARY"] >= 20]

# %% [markdown]
# We can also use more than one condition, but the syntax is strange (when compared to "straight" Python, although similar to NumPy):
#
# * Each condition must be surrounded by parentheses.
# * We use `&` for `and`, `|` for `or`, and `~` for `not`.
#
# For example, for if we want point guards (PG) that make over 12 million:

# %%
nba_by_player.loc[(nba_by_player["POSITION"] == "PG") & (nba_by_player["SALARY"] > 12)]

# %% [markdown]
# Here is a more complex (and contrived) example: suppose we want rows for players that either make less than 10 million, or play as a center, but not for the Chicago Bulls:

# %%
nba_by_player.loc[(nba_by_player["SALARY"] < 10) | 
                  ((~(nba_by_player["TEAM"] == "Chicago Bulls") & 
                    (nba_by_player["POSITION"] == "C"))
                  )]

# %% [markdown]
# The series methods [between](https://pandas.pydata.org/docs/reference/api/pandas.Series.between.html) and [isin](https://pandas.pydata.org/docs/reference/api/pandas.Series.isin.html) are also helpful when filtering.
#
# For example, for salaries above and including 10 million, but below (and excluding) 15, we can do: 

# %%
nba_by_player.loc[nba_by_player["SALARY"].between(10, 15, inclusive="left")]

# %% [markdown]
# The inclusion options for `.between` are:
#
# * `inclusive="both"` includes both, the *default*;
# * `inclusive="left"` includes left boundary only;
# * `inclusive="right"` includes right boundary only;
# * `inclusive="neither"` includes neither boundary.

# %% [markdown]
# If we want players that play for either the Chicago Bulls or New York Knicks:

# %%
nba_by_player.loc[nba_by_player["TEAM"].isin(["Chicago Bulls", "New York Knicks"])]

# %% [markdown]
# Another helpful series method for filtering is [str.contains](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.contains.html).  For instance, let's find all the John's in `nba`:

# %%
nba.loc[nba["PLAYER"].str.contains("John")]

# %% [markdown]
# Note that the search is case sensitive, but it does get last names starting with `"John"`.  We could add a space to the string, in this case:

# %%
nba.loc[nba["PLAYER"].str.contains("John ")]

# %% [markdown]
# We can also do it with `nba_by_player`, but we have to use the index:

# %%
nba_by_player.loc[nba_by_player.index.str.contains("John ")]

# %% [markdown]
# ### Filtering with `query`
#
# An alternative to filtering with `loc` is to use [query](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html).
#
# It's syntax is most often simpler, although somewhat strange.  It is somewhat less capable than `loc`, but it is often more convenient to use.
#
# `query` takes a *string* as its argument, and instead of specifying the column from the data frame, we can use simply the column's label in our conditions.  Moreover, we can use the more usual Python syntax, including the use of `and`, `or`, and `not`.
#
# Also note that `query` uses parentheses `( )` (as it is indeed a method), unlike `loc` (and `iloc`), which use square brackets `[ ]`.
#
# To illustrate its use, we rework the examples done using `loc` above here, now using `query`.
#
# To find the rows of centers:

# %%
nba_by_player.query("POSITION == 'C'")

# %% [markdown]
# Note that we do need quotes for `C`, but not for `POSITION`, since it is a column label.
#
# To find players who make 20 million or more:

# %%
nba_by_player.query("SALARY >= 20")

# %% [markdown]
# To find point guards that make over 12 million:

# %%
nba_by_player.query("POSITION == 'PG' and SALARY > 12")

# %% [markdown]
# To find rows for players that either make less than 10 million, or play as a center but not for the Chicago Bulls:

# %%
nba_by_player.query("SALARY < 10 or (not TEAM == 'Chicago Bulls' and POSITION == 'C')")

# %% [markdown]
# For salaries above and including 10 million, but below (and excluding) 15, we can do: 

# %%
nba_by_player.query("10 <= SALARY < 15")

# %% [markdown]
# For players that play for either the Chicago Bulls or New York Knicks:

# %%
nba_by_player.query("TEAM in ['Chicago Bulls', 'New York Knicks']")

# %% [markdown]
# The last example, where we find players named John, is one in which we have the rare case that it is a lot more complex with `query` than with `loc`, so we skip it here.

# %% [markdown]
# #### Spaces in Column Labels with `query`
#
# When using `query` we enter the column labels for filtering without quotes.  This poses a problem when there are spaces in the column label.  The solution is to surround the column label (inside the argument string) by back ticks `` ` ` ``.
#
# Let's rename the SALARY by YEARLY SALARY (just so we can illustrate the method) and filter for values less than 5:

# %%
nba_by_player.rename(columns={"SALARY": "YEARLY SALARY"}).query("`YEARLY SALARY` < 5")

# %% [markdown]
# #### Variables with `query`
#
# We cannot use values stored in a variable directly with `query`, since its argument is a string.  (It works fine with `loc`.)  But, we can do it by adding a `@` before the variable name in the argument string:

# %%
lower_bound_salary = 18  # you can change this value, without changing the next line

nba_by_player.query("SALARY >= @lower_bound_salary")

# %% [markdown]
# ## Extracting Single Entry
#
# We can extract a single entry with either `loc`, using labels, or with `iloc`, using positions:

# %%
nba

# %%
nba_by_player.loc["Jeff Teague", "TEAM"]

# %%
nba_by_player.iloc[3, 1]

# %% [markdown]
# Note that if we don't assign a column as the index, the index is numerical:

# %%
nba.loc[3, "TEAM"]

# %% [markdown]
# ## Dropping Columns
#
# We can use the method `drop` to remove columns.  We pass the column labels as the value for `columns=`:

# %%
nba_by_player.drop(columns="POSITION")

# %% [markdown]
# We can also drop more than one columns by passing a list:

# %%
nba_by_player.drop(columns=["POSITION", "TEAM"])

# %% [markdown]
# ## Adding Columns
#
# The recommended way to add a column is the use the [assign method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html).
#
# For instance, let's add an AGE columns to our `nba_by_player` data frame.  Here we will use fake, randomly generated ages.

# %%
ages = np.random.randint(18, 41, len(nba_by_player))  # random ages from 18 to 40

# %% [markdown]
# To add the array `ages` as the `AGE` columns, we do:

# %%
nba_by_player.assign(AGE=ages)

# %% [markdown]
# Note that we use the new column label as a *variable name*.
#
# Also, as all methods we've seen so far, `assign` returns a **new data frame, and does not change the original**:

# %%
nba_by_player

# %% [markdown]
# Of course, if we want to change the original, we can overwrite it as usual with
#
# ```python
# nba_by_player = nba_by_player.assign(AGE=ages)
# ```
#
# An alternative that adds the columns directly into the data frame (effectively changing it) is
#
# ```python
# nba_by_player["AGE"] = ages
# ```
#
# (Again, this last one *does* change the original `nba_by_player`!)
#
# We can also create more than one column with `assign`, by separating the assignments by commas:

# %%
heights = np.random.randint(72, 90, len(nba_by_player))  # random heights between 6' and 7'5"

nba_by_player.assign(AGE=ages, HEIGHT=heights)

# %% [markdown]
# Often we need to add a column added from some computation, but it is basically the same.  For instance, let's add a column "Difference from Average", that has the difference between players salary and the average salary from the data frame.
#
# A first obstacle here is that we have space in the column label, while `assign` uses a variable name, which cannot contain spaces.  So, we need to use a temporary label for the column and then rename it:

# %%
average_salary = nba_by_player["SALARY"].mean()

nba_by_player.assign(t=nba_by_player["SALARY"] - average_salary).rename(columns={"t": "Difference from Avarage"})

# %% [markdown]
# ## Columns Statistics
#
# We can get statistics for numerical columns with `.describe`:

# %%
nba_by_player.assign(AGE=ages, HEIGHT=heights).describe()

# %% [markdown]
# The percentages are the *quartiles*.  For instance, since the value of SALARY in the `25%` row is (about) $1.27$, this means that a quarter of the players make $1.27$ millions or less.   Since the value of SALARY in the `75%` row is $7$, this means that three quarters of the players make $7$ millions or less.
#
# `std` stands for *standard deviation*.

# %% [markdown]
# ## Columns Values
#
# For categorical columns, we can find which values are present by selecting it and using the `unique` method.
#
# So, if we want to see the list of teams in our data frame:

# %%
nba_by_player["TEAM"].unique()

# %% [markdown]
# ## `groupby`
#
# Suppose now we want to extra some more "indirect" information about the data.  For instance, suppose we would like to know what is the average salary paid *per team*.
#
# So, we want some information about a certain *category* in the data frame (like, team or position).  The tool for this sort of computation in pandas is [groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html).  It basically groups the data by the given category.
#
# When we group by team, in our example, we will have various values for the salary, and we need to tell pandas how to *aggregate* all these salaries for a single team.  In our case, we want the average, so we use the `mean` groupby method.
#
# On the other hand, since we cannot compute the average of the different values for the position, we need to first select only the columns we need, i.e., TEAM and SALARY in this case:

# %%
nba_by_player[["TEAM", "SALARY"]].groupby("TEAM").mean()

# %% [markdown]
# Note that the category we used to group by becomes the index of the resulting data frame.  
#
# If we have more columns from which we can use the same aggregating function (the average in this case), we can pass them all:

# %%
nba_by_player.assign(AGE=ages, HEIGHT=heights).drop(columns="POSITION").groupby("TEAM").mean()

# %% [markdown]
# Similarly, if we want the *median* salary per *position*:

# %%
nba_by_player[["POSITION", "SALARY"]].groupby("POSITION").median()

# %% [markdown]
# If we simply want to count how many rows we have for each category, we use the aggregating groupby method `size`.  In this case, there is no need to select/drop columns.
#
# For instance, to see how many players we have in each position:

# %%
nba_by_player.groupby("POSITION").size()

# %% [markdown]
# Note that the result is a *series*, and not a data frame.  We can make it into a data frame using the method `to_frame`:

# %%
nba_by_player.groupby("POSITION").size().to_frame()

# %% [markdown]
# Note that the label for the column is just the number 0.  We can specify the label by passing is (as a string) to the method:

# %%
nba_by_player.groupby("POSITION").size().to_frame("Count")

# %% [markdown]
# We can also use more than one aggregating function using the groupby method `agg`.  We pass a list of the aggregating methods we want to use *as strings*.  
#
# For instance, to see the minimum and maximum salary paid by a team:

# %%
nba_by_player[["TEAM", "SALARY"]].groupby("TEAM").agg(["min", "max"])

# %% [markdown]
# We can also group by more than one category, by passing a list as the argument.
#
# For instance, to see the average salary per team and position:

# %%
nba_by_player.groupby(["TEAM", "POSITION"]).mean()

# %% [markdown]
# The order matters:

# %%
nba_by_player.groupby(["POSITION", "TEAM"]).mean()

# %% [markdown]
# ## Pivot Tables
#
# Grouping by more than one category works well, but it can be a bit harder to visualize.  And alternative is to use *pivot tables*.  In this case, one of the two categories becomes the index (for rows) and the other becomes the columns.
#
# Here is how we would use it, similar to the previous example, so we can visualize the average salary per team and position:

# %%
nba_by_player.pivot_table(index="TEAM", columns="POSITION", values="SALARY", aggfunc="mean")

# %% [markdown]
# We can have more than a single value to aggregate:

# %%
nba_by_player.assign(AGE=ages, HEIGHT=heights).pivot_table(index="TEAM", columns="POSITION", values=["SALARY", "AGE"], aggfunc="mean")

# %% [markdown]
# Note the labels over the columns, specifying the values that were aggregated.

# %% [markdown]
# Note also that we have some strange entries above: `NaN`.  It stands for "not a number", and corresponds to entries that are not available.  For instance, the `NaN` in the Miami Heat row and C columns indicates that there are no centers in the Miami Heat team:

# %%
nba_by_player.query("TEAM == 'Miami Heat'")

# %% [markdown]
# ## Chaining
#
# Often we want to perform a series of operations on a data frame.  This is often done by *chaining* the methods we need.  (We've already seen some examples above!)
#
# Let's do one more example, which is again contrived, but illustrates the idea.  Suppose we want an array with the five teams that pay their players under 30 the most on average, in order of paying the best to worst.  (Note that we need to manually add the AGE column.)
#
# Here is how it can be done:

# %%
nba_by_player.assign(AGE=ages).query("AGE < 30")[["TEAM", "SALARY"]].groupby("TEAM").mean().sort_values("SALARY", ascending=False).index.to_numpy()[:5]

# %% [markdown]
# Note that the line is too long.  Unfortunately, since white space matters in Python, we cannot just break lines.  On the other hand, we can if we surround the command by parentheses:

# %%
(
    nba_by_player.assign(AGE=ages)  # add age column
    .query("AGE < 30")  # select under 30
    [["TEAM", "SALARY"]]  # select desired columns
    .groupby("TEAM").mean()  # groupby by team and take average
    .sort_values("SALARY", ascending=False)  # sort
    .index  # extract the index
    .to_numpy()  # make it into an array
    [:5]  # first 5
)

# %% [markdown]
# Note that breaking the lines allows us to add comments as well! 

# %% [markdown]
# ## Some Useful Options
#
# There are two options that we can use to improve pandas' default behavior.  These should come after
#
# ```python
# import pandas
# ```
#
# The first one is called [Copy on Write](https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html).  Although it should not affect us in this series of lectures, it helps avoid some cryptic error messages and unexpected behaviors.  (More details on the provided link.)  You the configuration is done with the code below:

# %%
pd.options.mode.copy_on_write = True

# %% [markdown]
# The second option improves the way pandas deals with strings, providing significant improvements when dealing with string objects in data frames.  You can set it with the code below:

# %%
pd.options.future.infer_string = True

# %% [markdown]
# **Important:** These options might not work if your version of pandas is relatively old.  In that case, you might get errors running the two code blocks above.  For this reason, these lines will be commented out in the following notebooks of this series, but if the lines above do not give you error, it is recommended that you always use them.

# %% [markdown]
# **Note:** These two options will become the default behavior for pandas 3.0 and later.

# %% [markdown]
# ## Comments, Suggestions, Corrections
#
# Please send your comments, suggestions, and corrections to lfinotti@utk.edu.
