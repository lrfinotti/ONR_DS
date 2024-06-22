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
# # NumPy
#
# ## Introduction
#
# [NumPy](https://numpy.org/) is a Python library for fast (numerical) computations with arrays.  We use NumPy arrays instead of Python's lists when we need to perform fast computations with the encapsulated data.
#
# NumPy does not come with Python, so it needs to be installed separately.  If you have a *vanilla* installation of Python, you can do it by running 
#
# ```
# pip install numpy
# ```
#
# from a terminal.
#
# On the other hand, if you installed Anaconda, it should already be available.

# %% [markdown]
# After installation, we need to import it.  We usually give is the shortcut `np`, so that we can call its commands with `np.function` instead of `numpy.function`.  (This shortcut is the standard.)

# %%
import numpy as np

# %% [markdown]
# ## Creating Arrays

# %% [markdown]
# ### Manual Creation
#
# We can create a NumPy array (which I will refer to simply as an array from now on), by converting a Python list, using the function `np.array`:

# %%
first_array = np.array([1, 2, 3, 4])

first_array

# %%
second_array = np.array([2.41, 3.11, 5.7, 11.0])

second_array

# %% [markdown]
# ### Ranges
#
# We also have `np.arange`, similar to Python's `range`, to create arrays following a pattern:

# %%
np.arange(10)

# %%
np.arange(3, 15)

# %%
np.arange(4, 32, 5)

# %% [markdown]
# We can also create an array with a predetermined number of entries *equally spaced* between given first and last elements using `np.linspace`:

# %%
np.linspace(1, 10, 25)  # array with 25 elements, starting at 1 and ending at 10

# %% [markdown]
# ### Repetitions
#
# We can create arrays of zeros and ones of a specified type as well, with `np.zeros` and `np.ones`, respectively:

# %%
np.zeros(10)  # 10 zeros (floats by default)

# %%
np.zeros(10, dtype=np.int64)  # 10 integer zeros

# %%
np.zeros(10, dtype=bool)  # the boolean zero is False

# %%
np.ones(5)  # 5 ones (floats by default)

# %%
np.ones(5, dtype=np.int64)  # 5 integer zeros

# %%
np.ones(5, dtype=bool)  # the boolean one is True

# %% [markdown]
# To create an array that repeats the same element, we can use `np.repeat` or `np.full`:

# %%
np.repeat(2, 7)  # seven twos

# %%
np.full(7, 2)  # seven twos again -- note the parameters are switched!

# %% [markdown]
# If we pass an array instead of an element to `np.repeat`, it repeats each element:

# %%
np.repeat(np.array([1, 2, 3]), 4)

# %% [markdown]
# To repeat in sequence, we can use `np.tile`:

# %%
np.tile(np.array([1, 2, 3]), 4)

# %% [markdown]
# ### Random Arrays
#
# There are a few ways to create random arrays.  (Functions for generating random data in NumPy start with `np.random`.)
#
# To create an array with a given number of random floats between 0.0 and 1.0:

# %%
np.random.random(6)  # 6 random numbers between 0.0 and 1.0

# %% [markdown]
# (Each time you run the cell above, you will get a different array.)

# %% [markdown]
# To create an array with a given number of integers in a range:

# %%
np.random.randint(1, 6, 20)  # 20 random numbers between 1 and 5 (NOT 6)

# %% [markdown]
# We can also created numbers randomly by with probabilities from a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) with a given average and standard deviation using `np.random.normal`.
#
# For 20 random numbers chosen from with a normal probability curve with average 10 and standard deviation 2:

# %%
np.random.normal(10, 2, 20)

# %% [markdown]
# To select random numbers from a list, we can use `np.random.choice`:

# %%
np.random.choice(np.array([1, 3, 7]), 10)

# %% [markdown]
# ### Summary
#
# | **Function**  | **Description**                                          |
# |---------------|----------------------------------------------------------|
# | `np.array`    | Converts list to array                                   |
# | `np.arange`   | Creates array range of numbers                           |
# | `np.linspace` | Creates equally spaces numbers between given boundaries  |
# | `np.zeros`    | Create array of zeros (specify type with `dtype=<type>`) |
# | `np.ones`     | Create array of ones (specify type with `dtype=<type>`)  |
# | `np.full`     | Create array of with a single element repeated           |
# | `np.repeat`   | Create an array repeating each element of given array    |
# | `np.tile`     | Create an array repeating array (in order)               |

# %% [markdown]
# ## Length
#
# The function `len` can also be used to get lengths of arrays:

# %%
first_array

# %%
len(first_array)

# %% [markdown]
# For *one-dimensional array* (so arrays that do not have other arrays as elements), we can also get the length of an array from the parameter `.size`:

# %%
first_array.size

# %% [markdown]
# (Since `.size` is a parameter and not a method/function, we *cannot use parentheses*!)

# %% [markdown]
# ## Operations with Arrays
#
# If we perform an operation between a number and an array, it will perform the operation between the number and every single entry of the array:

# %%
first_primes = np.array([2, 3, 5, 7])

first_primes

# %%
3 + first_primes

# %%
3 * first_primes

# %%
first_primes / 2

# %%
first_primes == 3

# %% [markdown]
# If we have two arrays of the *same size*, operations between them are performed componentwise:

# %%
first_squares = np.array([0, 1, 4, 9])

first_squares

# %%
first_primes + first_squares

# %%
first_primes * first_squares

# %%
first_primes ** first_squares

# %%
first_primes > first_squares

# %% [markdown]
# ## Some NumPy Functions
#
# NumPy provides many functions for mathematical computations and array manipulation.  The mathematical functions are optimized for computation in arrays.  (It is *much* faster than performing the same computation individually in each entry of the array.)
#
# For example:

# %%
np.sqrt(first_squares)  # square root

# %%
np.log(first_primes)  # natural log

# %%
np.sin(np.pi * first_primes / 8 + np.pi / 3)  # sine function -- np.pi is the number pi

# %% [markdown]
# Here are some other useful functions:
#
# Each of these functions takes an array as an argument and returns a single value.
#
# | **Function**       | Description                                                          |
# |--------------------|----------------------------------------------------------------------|
# | `np.prod`          | Multiply all elements together                                       |
# | `np.sum`           | Add all elements together                                            |
# | `np.mean`          | Average of all elements                                              |
# | `np.median`        | Median of all elements                                               |
# | `np.std`           | Standard deviation of all elements                                   |
# | `np.max`           | Maximum of all elements                                              |
# | `np.min`           | Minimum of all elements                                              |

# %% [markdown]
# See also the [full Numpy reference](http://docs.scipy.org/doc/numpy/reference/).

# %% [markdown]
# ## Slicing
#
# Element extraction and slicing works just as with lists:

# %%
more_primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

more_primes

# %%
more_primes[3]  # fourth element

# %%
more_primes[-2]  # second to last element

# %%
more_primes[2:-3]

# %%
more_primes[:5]

# %%
more_primes[4:]

# %%
more_primes[1:8:2]

# %% [markdown]
# ## Filtering
#
# We can select entries of an array by giving a boolean array (or list) with `True` in the positions we want to keep and `False` in the positions we want to drop:

# %%
a = np.arange(1, 6)
a

# %%
mask = [True, False, False, True, True]

mask

# %%
a[mask]

# %% [markdown]
# This allows us to filter lists by conditions.  For instance, to filter the array `more_primes` for those which are less than 10, we can do:

# %%
more_primes[more_primes < 10]

# %% [markdown]
# ## Counting
#
# If we want to count how many elements of `more_primes` are greater than 7, we could do:

# %%
len(more_primes[more_primes > 7])

# %% [markdown]
# Alternatively, we can use `np.count_nonzero`.  As the name says, it counts the number of non-zero elements in an array:

# %%
np.count_nonzero(np.array([0, 1, 2, 0, 4, 0, 3]))

# %% [markdown]
# On the other hand, in Python the boolean `False` is (sometimes) treated as zero, while `True` is non-zero.  Hence, `np.count_nonzero` can be used to count the number of `True`'s in a boolen array:

# %%
np.count_nonzero(np.array([True, True, False, False, True]))

# %% [markdown]
# Thus, we could also have done (in the previous example):

# %%
np.count_nonzero(more_primes > 7)

# %% [markdown]
# We can also apply this idea to count how many elements match between two arrays:

# %%
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([1, 4, 3, 2, 5])

np.count_nonzero(array1 == array2)

# %% [markdown]
# ## Equality of Arrays
#
# We cannot check if two arrays are equal using `==`, as it performs the comparison componentwise:

# %%
array1 == array2

# %% [markdown]
# Instead, we must use `np.array_equal`:

# %%
np.array_equal(array1, array2)

# %%
np.array_equal(array1, np.array([1, 2, 3, 4, 5]))

# %% [markdown]
# ## Values and Counts
#
# It is easy to see how many times a single value occurs in an array using the tools we've already seen.
#
# For instance, if we have

# %%
a = np.random.choice(np.arange(5), 10)
a

# %% [markdown]
# and we want to count how many threes we have, we can simply do:

# %%
np.count_nonzero(a == 3)

# %% [markdown]
# But, sometimes we want to see how many occurrences we have for *each value* of the array, not just a single value.  For that, we can use [np.unique](https://numpy.org/doc/stable/reference/generated/numpy.unique.html).  
#
# By default, it simply gives the unique values in the array, *sorted*.  This can be useful when we are not sure what values actually occur in a array, e.g., when we create an array randomly:

# %%
# 10 values between 0 and 99
b = np.random.randint(0, 100, 10)
b

# %%
np.unique(b)

# %% [markdown]
# With our previous array `a`:

# %%
np.unique(a)

# %% [markdown]
# To actually get how many times each unique value occurs, we need the optional argument `return_counts=True`

# %%
np.unique(a, return_counts=True)

# %% [markdown]
# The output is now two arrays: the first one is the same, the unique values *sorted*, while the second array contains the counts for each *respective* unique value.
#
# As another example:

# %%
# 100 values between 0 and 9
c = np.random.randint(0, 10, 100)

values, counts = np.unique(c, return_counts=True)

for value, count in zip(values, counts):
    print(f"The value {value} occurs {count:>2} times.")

# %% [markdown]
# ## Arrays of Strings
#
# We can also perform operations with arrays of strings:

# %%
string_array = np.array(["A", "bb", "CcC", "ddDD"])
string_array

# %% [markdown]
# NumPy functions for strings start with `np.char`.
#
# For instance, to convert to lower case:

# %%
np.char.lower(string_array)

# %% [markdown]
# To convert to upper case:

# %%
np.char.upper(string_array)

# %% [markdown]
# We can replace occurrences of `C` with `X` in the strings:

# %%
np.char.replace(string_array, "C", "X")

# %% [markdown]
# ### Other Functions
#
# Here are some useful functions for strings,
#
# Each of these functions takes an array of strings and returns an array.
#
# | **Function**        | **Description**                                              |
# |---------------------|--------------------------------------------------------------|
# | `np.char.lower`     | Lowercase each element                                       |
# | `np.char.upper`     | Uppercase each element                                       |
# | `np.char.strip`     | Remove spaces at the beginning or end of each element        |
# | `np.char.isalpha`   | Whether each element is only letters (no numbers or symbols) |
# | `np.char.isnumeric` | Whether each element is only numeric (no letters)  |
#
#
# Each of these functions takes both an array of strings and a *search string*; each returns an array.
#
# | **Function**         | **Description**                                                                  |
# |----------------------|----------------------------------------------------------------------------------|
# | `np.char.count`      | Count the number of times a search string appears among the elements of an array |
# | `np.char.find`       | The position within each element that a search string is found first             |
# | `np.char.rfind`      | The position within each element that a search string is found last              |
# | `np.char.startswith` | Whether each element starts with the search string                               |
#
# You can find a lot more about string functions with `help(np.char)`.

# %% [markdown]
# ## Examples

# %% [markdown]
# ### Example: Converting Temperatures
#
# As a simple application, let's convert an array of temperatures in Fahrenheit to Celsius.
#
# First, let's create a randomize array of temperatures that between $T_0 - \Delta$ and $T_0 + \Delta$, where $T_0$ is some base temperature and $\Delta$ is some temperature variation.  (Let's set them to $60$ and $40$ respectively.)

# %%
base_temperature = 60  # T_0
variation = 40  # Delta
number_of_temps = 30

temperatures = base_temperature - variation + 2 * variation * np.random.random(number_of_temps)

temperatures

# %% [markdown]
# The formula to convert to Celsius is:
#
# $$
# \text{Temp in Celsius} = \frac{5}{9} \cdot \left( \text{Temp in Fahrenheit} - 32 \right)
# $$
#
# So, we can covert with:

# %%
temperatures_celsius = 5 / 9 * (temperatures - 32)

temperatures_celsius

# %% [markdown]
# We can round to two decimal places with `np.round`:

# %%
temperatures_celsius = np.round(5 / 9 * (temperatures - 32), 2)

temperatures_celsius

# %% [markdown]
# ### Example: Checking a Trigonometric Identity
#
# It is a well-know theorem that for any real number $x$, we have that 
#
# $$
# \cos^2(x) + \sin^2(x) = 1.
# $$
#
# Let's check this with some concrete examples!
#
# We first create a large set of numbers to try.  Since the values of sine and cosine are periodic of period $2 \pi$ (in other words, $\cos(x + 2\pi) = \cos(x)$ and $\sin(x + 2\pi) = \sin(x)$), we can simply check it for several numbers between $0$ and $2\pi$.

# %%
number_of_tests = 10 ** 5  # one hundred thousand tests!
test_cases = 2 * np.pi * np.random.random(number_of_tests)

# %% [markdown]
# Now we compute the sine and cosines, square them, and then add them:

# %%
result = np.cos(test_cases) ** 2 + np.sin(test_cases) ** 2

# %% [markdown]
# If the theorem is really true, we should get `number_of_test` ones in `result`.  So, let's check:

# %%
np.count_nonzero(result == 1)

# %% [markdown]
# Oh-oh, something is not quite right.  Either our theorem is not true, or there is something fishy here.
#
# Let's inspect the first 10 elements for which we do not get one:

# %%
result[result != 1][:10]

# %% [markdown]
# What?  They seem to be one...  Do we need 1.0 (float) instead of 1 (int)?

# %%
np.count_nonzero(result == 1.0)

# %%
result[result != 1.0][:10]

# %% [markdown]
# Apparently not.  *The reason is that floats have approximation errors, so it is hard to tell when two floats are actually equal!*
#
# So, let's simply check if the result is really *close* to one, say the difference less than $10^{-6} = 0.000001$:

# %%
margin_of_error = 10 ** (-9)
np.count_nonzero(np.abs(result - 1) < margin_of_error)

# %% [markdown]
# Ah, so *all* of the results were within our margin of error.

# %% [markdown]
# ### Example: Leibniz's formula for $\pi$
#
# **Acknowledgment:** This is based in an example from [Computational and Inferential Thinking: The Foundations of Data Science](https://inferentialthinking.com/chapters/intro.html), by A. Adhikari, J. DeNero, D. Wagner.
#
# [Gottfried Wilhelm Leibniz](https://en.wikipedia.org/wiki/Gottfried_Wilhelm_Leibniz) 
# (1646 - 1716) discovered the following formula for $\pi$ as an infinite sum of fractions:
#
# $$\pi = 4 \cdot \left(1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \frac{1}{9} - \frac{1}{11} + \dots\right)$$
#
# (For the math inclined: this is related to the fact that $\arctan(1) = \pi/4$.  You can then use the [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) for $\arctan$.)
#
# By stopping after a finite number steps we get an *approximation* of $\pi$, with better approximations for larger number of terms.  Let's check this approximation using the first $5{,}000$ terms!
#
# $$4 \cdot \left(1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \frac{1}{9} - \frac{1}{11} + \dots + \frac{1}{9997} - \frac{1}{9999} \right)$$
#
# We will add the postiive terms, subtract the negative terms, and then multiply by $4$:
#
# $$4 \cdot \left( \left(1 + \frac{1}{5} + \frac{1}{9} + \dots + \frac{1}{9997} \right) - \left(\frac{1}{3} + \frac{1}{7} + \frac{1}{11} + \dots + \frac{1}{9999} \right) \right)$$
#
# Let's start by making an array with the positive denominators:

# %%
positive_term_denominators = np.arange(1, 9998, 4)

# %% [markdown]
# To get the terms, we need to invert them:

# %%
positive_terms = 1 / positive_term_denominators

# %% [markdown]
# We could something similar for the negative terms, but it is a bit simpler:

# %%
negative_term_denominators = 2 + positive_term_denominators
negative_terms = 1 / negative_term_denominators

# %% [markdown]
# Now we can just add the terms, subtract the two, and multiply by $4$:

# %%
leibniz_pi = 4 * (np.sum(positive_terms) - np.sum(negative_terms))
leibniz_pi

# %% [markdown]
# Let's compare it to the numerical approximation of $\pi$ from NumPy:

# %%
np.pi

# %%
np.pi - leibniz_pi

# %% [markdown]
# ## Efficiency
#
# ### Avoid Loops
#
# The "usual" way to compute something like that would be to use loops.  Note that we have:
#
#
# $$\pi = 4 \cdot \left(1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \frac{1}{9} - \frac{1}{11} + \cdots + (-1)^i \frac{1}{2i + 1} + \cdots \right)$$
#
# where $(-1)^i \dfrac{1}{2i + 1}$ is the $(i + 1)$-th term of the sum.  (The sum starts with $i=0$.)
#
# Let's compute it with a loop (as would be natural):

# %%
result = 0
for i in range(5000):
    sign = (-1) ** i
    result += sign / (2 * i + 1)
4 * result

# %% [markdown]
# This, of course, it also works!  But let's time it:

# %%
# %%timeit
res = 0
for i in range(5000):
    sign = (-1) ** i
    res += sign / (2 * i + 1)
4 * res

# %% [markdown]
# Now, let's look at the time for the computation using NumPy (i.e., "vectorized" computations):

# %%
# %%timeit
positive_denominators = np.arange(1, 9998, 4)
4 * ((1 / positive_denominators).sum() - (1 / (positive_denominators + 2)).sum())

# %% [markdown]
# As you can see, the second method is *a lot* more efficient!  
#
# On the other hand, the second memory uses more memory, as it has to create large arrays.  (In my test it used over 6 times the amount of memory as the first, which uses very little memory.  So, it is still not bad!)
#
# And of course, even though the first method was about 50 times slower, it makes virtually no difference in this example!  But it does make a difference in more complex examples.
#
# So: **avoid using loops and use NumPy's vectorized functions instead whenever possible!**

# %% [markdown]
# ### Do Not Append to Array
#
# Here is another common mistake: although we can add elements to arrays using `np.append`, this operation is relatively slow.  Instead, try to initialize your array with zeros (of the right type), and then override its values.
#
# For instance: suppose you want to choose ten random values between 0.0 and 1.0, find the average and add the result to an array *one hundred thousand times*.  We could do:

# %%
# %%time

number_of_repetitions = 100_000

results = np.array([])

for i in range(number_of_repetitions):
    results = np.append(results, np.mean(np.random.random(10)))

results[:10]

# %% [markdown]
# Instead, it is a lot more efficient to initialize the an array to the correct size (with zeros, for instance) and data type, and then overwrite the entries.  

# %%
# %%time

number_of_repetitions = 100_000

results = np.zeros(number_of_repetitions)

for i in range(number_of_repetitions):
    results[i] = np.mean(np.random.random(10))

results[:10]

# %% [markdown]
# As one can clearly see, this second method is a lot faster!

# %% [markdown]
# If you are familiar with [list comprehensions](https://docs.python.org/2/tutorial/datastructures.html#list-comprehensions) in Python, the previous method is just as efficient as:

# %%
# %%time

number_of_repetitions = 100000

results = np.array([np.mean(np.random.random(10)) for i in range(number_of_repetitions)])

results[:10]

# %% [markdown]
# ## More Examples

# %% [markdown]
# ### Example: Computing Grades
#
# We've only used *one-dimensional* arrays so far.  Now suppose we have an array with arrays as its elements, a *two-dimensional array*.  Each entry is an array with grades for a student.
#
# For example, lets create six grades for ten students (so an array with 10 arrays of 6 grades each) with normal distribution of average $80$ and standard deviation $15$.  (We might get grades over $100$.  Let's just assume that they were obtained by extra-credit.)

# %%
grades = np.array([np.round(np.random.normal(80, 15, 6)).astype(int) for i in range(10)])

grades

# %% [markdown]
# Suppose that the first three grades are homework grades, with weight of $10\%$ each, the next two are midterm grades, with weights of $20\%$ each, and the last is the final grade, with weight $30\%$.  
#
# **Goal:** We want to compute the students averages.
#
# First, let's create an array with the weights:

# %%
weights = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3])

# %% [markdown]
# Now, we want to multiply each grade by its weight.  Since the arrays in `grades` have the same length as `weight`, we can do it directly!

# %%
weighted_grades = grades * weights

weighted_grades

# %% [markdown]
# Now we have to add all the grades for each student.  We can try `np.sum`:

# %%
np.sum(weighted_grades)

# %% [markdown]
# That is not what we want!  It added all grades, from all students, together!
#
# We want to just add the arrays in `weighted_grades`.  We can do that by specifying the argument `axis=1`.  (In reality `axis=1` implies we are adding entries from a row.  Using `axis=0` would add entries in a column.)

# %%
averages = np.sum(weighted_grades, axis=1)

averages

# %% [markdown]
# (As expected, the averages are pretty close to $80$.)

# %% [markdown]
# #### Dropping a Homework Grade
#
# What if I want to drop the lowest homework score?  
#
# Then, the weight of each homework grade must be $15\%$ instead of $10\%$, so let's adjust the weights:

# %%
new_weights = np.array([0.15, 0.15, 0.15, 0.2, 0.2, 0.3])

new_weights

# %% [markdown]
# We can again get weighted grades by simply multiplying.

# %%
new_weighted_grades = grades * new_weights

# %% [markdown]
# Now, we want to add the grades, but drop the lowest homework score.  So, let's start my getting simply the weighted homework grades by *slicing* `weighted_grades`:

# %%
new_weighted_grades

# %% [markdown]
# We want (all rows and) the first three columns.  In a two-dimensional array, we can get slices of rows and columns, giving them *in this order* (slice for the rows first, then slice for the columns), separated by a comma:

# %%
new_weighted_hws = new_weighted_grades[:, :3]  # all rows [:], and first three columns: [:3]

new_weighted_hws

# %% [markdown]
# Now, I need the minimum *of the rows*.  If I simply use `np.min`:

# %%
np.min(new_weighted_hws)

# %% [markdown]
# Again, it computes the minimum of all entries in the array.  And again, we solve this by using the optional argument `axis=1`, just like with `np.sum`:

# %%
min_hw_scores = np.min(new_weighted_hws, axis=1)

min_hw_scores

# %% [markdown]
# Now, I can simply subtract this lowest weighted homework score from the sum of all weighted grades:

# %%
new_averages = np.sum(new_weighted_grades, axis=1) - min_hw_scores

new_averages

# %% [markdown]
# ### Example: Compound Interest
#
# Let's now create an array that has the monthly balances from a savings account.
#
# If
#
# $$
# \begin{align*}
#   P &= \text{principal (or initial amount)}, \\
#   r &= \text{interest rate (APR) as a percent}, \\
#   t &= \text{number of days}, \\
#   F &= \text{final value (after $t$ days)},
# \end{align*}
# $$
#
# then, starting with $P$ dollars, in a savings account with rate of $r$, after $t$ days, we will have
#
# $$
# F = P \cdot \left( {1 + \frac{r}{100 \cdot 365}} \right)^t
# $$
#
# Let's assume we have $2{,}000$ dollars in an account with $3\%$ rate.

# %%
P = 2000  # principal (initial amount)
r = 3  # interest rate (in %)

# %% [markdown]
# As will see, it will also be handy to save
#
# $$
# 1 + \frac{r}{100 \cdot 365}
# $$
#
# in a variable, which we will call `factor`:

# %%
factor = 1 + r / (100 * 365)

# %% [markdown]
# Now, lets make an array with the (approximate) monthly balances for the next $3$ years (so $36$ months).  Let's round the number of days in a month to $30$.
#
# We can start by creating an array with
#
# ```python
# [1, factor ** 30, factor ** 60, factor ** 90, ..., factor ** 1080]
# ```
#
# as then we only need to multiply this array by $P$.
#
# First, though, we create the exponents
#
# ```python
# [0, 30, 60, 90, ... , 1080]
# ```
#
# To make the code more flexible, let's use a variable for the number of months:

# %%
months = 36

# %% [markdown]
# Now the exponents:

# %%
exponents = np.arange(0, months * 30 + 1, 30)  # NOTE THE "+1" to include the last term!

exponents

# %% [markdown]
# Then, we produce
#
# ```python
# [1, factor ** 30, factor ** 60, factor ** 90, ..., factor ** 1080]
# ```

# %%
factors = factor ** exponents

factors

# %% [markdown]
# Now, we just multiply by $P$ (rounding to 2 decimals):

# %%
balances = np.round(P * factors, 2)

balances

# %% [markdown]
# So, after $10$ months, the balance is:

# %%
balances[10]

# %% [markdown]
# ####  Adding Deposits
#
# Let's now assume that we deposit the same amount very month (i.e., every $30$ days) in our savings account. Let's call this amount $A$ and choose, say, $\$150$:

# %%
A = 150

# %% [markdown]
# The problem now is that I cannot just add $A$ every month, since after that money is in our account, it also affects the interest paid!
#
# What we need to add to `balances` is in fact:
#
# ```python
# [0, 
#  A, 
#  A + A * factor ** 30, 
#  A + A * factor ** 30 + A * factor ** 60, 
#  A + A * factor ** 30 + A * factor ** 60 + A * factor ** 90, ...].
# ```
#
# so we account for the interest for each deposit.
#
# Let's start by producing
#
# ```python
# [A, 
#  A * factor ** 30,
#  A * factor ** 60, 
#  A * factor ** 90, ...]
# ```
#
# with length equal to the total number of months *minus 1*.  Let's again round as well:

# %%
deposits = np.round(A * factors[:-1], 2)

deposits

# %% [markdown]
# Now we want to go from our `deposits`
#
# ```python
# [A, 
#  A * factor ** 30, 
#  A * factor ** 60, 
#  A * factor ** 90, ...]
# ```
#
# to
#
# ```python
# [A, 
#  A + A * factor ** 30, 
#  A + A * factor ** 30 + A * factor ** 60, 
#  A + A * factor ** 30 + A * factor ** 60 + A * factor ** 90, ...].
# ```
#
# That is simply the *cumulative sum* of the array!  The cumulative sum of $(x_1, x_2, x_3, x_4)$ is simply
#
# $$
# \begin{align*}
#  (&x_1, \\
# &x_1 + x_2, \\
# &x_1 + x_2 + x_3\\
# &x_1 + x_2 + x_3 + x_4).
# \end{align*}
# $$
#
# Fortunately NumPy can do that for us with the function [np.cumsum](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html).  
#
# For instance `np.cumsum(np.array([1, 2, 3, 4]))` gives `[1, 3, 6, 10]`.

# %%
deposits = np.cumsum(deposits)

deposits

# %% [markdown]
# Finally, we need to add a zero at the beginning and add the results to our previous `balances`:

# %%
deposits = np.append(0, deposits)

deposits

# %%
new_balances = balances + deposits

new_balances

# %% [markdown]
# Now, after $10$ months, the balance is:

# %%
new_balances[10]

# %% [markdown]
# ## Comments, Suggestions, Corrections
#
# Please send your comments, suggestions, and corrections to lfinotti@utk.edu.
