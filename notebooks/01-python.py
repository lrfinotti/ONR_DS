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
# # Introduction Python and Jupyter Lab (for Data Science)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Introduction
#
# In this notebook we introduce the very basics of the [Python](https://www.python.org/) programming language and [Jupyter Notebooks](https://jupyter.org/).
#
# We will only discuss the basics and still rather fast, but hopefully it will be enough to get students started.
#
# We will focus on (basic) tools that are most commonly used in applications to data science.

# %% [markdown]
# ## Python
#
# [Python](https://www.python.org/) is a simple yet powerful programming language.
#
# Some of its good qualities:
#
# * **Easy syntax:** it is easy to learn, and quick to write programs.
# * **Extensible:** there are thousands of modules/libraries that make it easy to accomplish most tasks.
# * **Resources:** due to its popularity, it is very easy to find help on how to do something in Python.  (In particular, [Stack Overflow](https://stackoverflow.com/) (which contains Q&A) and [YouTube](https://www.youtube.com/) contain extensive content on Python.)
# * **Free and Open Source:** there is no cost to use it, it's been continually improved, and has a strong community behind it.
#
# Some of its drawbacks:
#
# * **Relatively slow:** it cannot compete with some languages, such as C/C++, Java, and Rust in speed.
# * **"Quirky":** it sometimes does things differently than other languages.
#
# In *many* cases, the "slow" problem is not a real problem: would you rather spend one hour writing code that will run in 0.1 millisecond, or 10 minutes writing code that will run in 10 milliseconds?  Computers today are so fast that for most "mundane" tasks, Python will be more than fast enough, and considerably easier to use.
#
#
# ### Python for Data Sciences
#
# Python became popular for data science due to its ease of use (and thus gentle learning curve).  But since data science often requires *a lot* of computational power, it seems that it might not be the best option.  This indeed would be the case if were not for specialized libraries that allow Python to perform specific kinds of computations (including the ones we need in data science) almost as fast as faster (and more complex) languages! 
#
# More specifically, [NumPy](https://numpy.org/) allows us to perform computations with arrays of data extremely fast, and it is used by most other packages/libraries that require such computations, such as [pandas](https://pandas.pydata.org/), one of the main libraries for data science.  (We will not discuss those in this notebook, though.)

# %% [markdown]
# ## Jupyter Notebooks
#
# [Jupyter Notebooks](https://jupyter.org/) allow us to create interactive documents containing code along  with properly formatted text, and thus is quite useful for teaching, presentations, and documenting code.
#
# This is a Jupyter notebook!
#
# Below we have an example of a *code cell*, which runs Python code:

# %%
print("Hello worlds!")

# %% [markdown]
# (You can run other languages in a Jupyter notebook, but here will stick with Python only.)

# %% [markdown]
# You can edit the code, by clicking in *cell* containing and editing it, and run the code by pressing `Shift + Enter` in your keyboard.  It will also select the next cell (code or text). If there is no cell following it, it will create a new one.
#
# You can also edit the text you see here: just double click on the text (the cell will enter in *Edit mode* and will look a bit different and will not be formatted), make the necessary changes, and then press `Shift + Enter`.

# %% [markdown]
# ### Working with Jupyter Notebooks
#
# There are different software that can load and run Jupyter notebooks.  I recommend [Jupyter Lab](https://jupyter.org/) and will describe here how its interface works with Jupyter notebooks.  (Other software that also run Jupyter notebooks often behave similarly.)  Jupyter Lab runs on your browser.
#
# **Creating a new notebook:** You can create a new notebook by clicking on the plus symbol (`+`) on the top of the Jupyter Lab window.  Then, click on the Python icon under *Notebook* (on top).
#
# **Open an existing notebook:** After launching Jupyter Lab, you can open saved notebooks (which are files with extension `ipynb` by default) by clicking on the folder icon on the left pane, navigating to the file, and then double-clicking on the desired file.
#
# **The main menu:** You will find a familiar menu on top, from which you can find most operations, such as saving, closing, shutting down, copying/cutting/pasting, etc.
#
# **Notebook toolbar:** Under the tab with the name of the notebook you will find icons for: 
# * saving the notebook, 
# * adding a cell (below current), 
# * cut, copy, paste cells, 
# * run cell (as an alternative to pressing `Shift + Enter`), 
# * interrupt kernel (which stops a computation that might be taking to long), 
# * restart kernel (which will give a fresh start for the notebook --- code cells need to be run again to take effect), 
# * restart kernel and run all cells, 
# * a drop down menu to choose the cell type between between *Markdown* (used for text), *Code*, and *Raw*.

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Notebook Modes
#
# A notebook has two modes: *edit mode* or *command mode*.  You are in edit mode when you are editing a cell.  (Remember, you can edit a code cell by simply clicking on it, and a text cell by *double*-clicking on it.)
#
# In edit mode, you will always see the blinking cursor that allows you to enter text.  You can exit edit mode (and thus enter command mode) by pressing the `Escape` key.
#
# In command mode, you will see no blinking cursor, but you will see a blue vertical line to the left of the current selected cell.  In this mode you can issue keyboard commands.  For example:
#
# | **Key**                | **Command**                                |
# |------------------------|--------------------------------------------|
# | `A`                    | Add cell *above* selected one              |
# | `B`                    | Add cell *below* selected one              |
# | `M`                    | Change selected cell to Markdown/Text cell |
# | `Y`                    | Change selected cell to Code cell          |
# | `C`                    | Copy selected cells                        |
# | `X`                    | Cut selected cells                         |
# | `V`                    | Paste copied/cut cells below selected cell |
# | `Z`                    | Undo                                       |
# | `DD` (press `D` twice) | Delete selected cells                      |
#
#
# (Note that we follow the convention that `A` represents that key in the keyboard, so *without pressing the `Shift` key*.  We would write `Shift + A` if `Shift` needs to be pressed.)
#
# ### Cells
#
# A Jupyter notebook is divided in cells.  We have (basically) two types of cells: text and code.
#
# Code cells have a different darker background and have square brackets `[ ]` on its left, often with an number.  The number represents the order in which the code cell was run.
#
# You can run Python code in code cells.  The code us evaluated by running it (with `Shift + Enter` or clicking on the "Run" icon in the notebook menu).  If the last line of code in the cell produces any output, this result is printed below the cell after running it.  (Note that *only the output of the **last line** is printed*.)
#
# Text cells cannot immediately be seen, as they simply appear as text.  If we have many text cells together, it is not clear at first where one ends and the other begins.  But if you click on a text cell, a blue line appears to its left.
#
#
# #### Changing Cell Types
#
# If you create a new cell by clicking on the `+` in the notebook menu (below the tab with the notebook file name) or by pressing `A` or `B` in command mode.  By default it will be a code cell, but you can change it either by pressing `M` in command mode (pressing `Escape` to enter command mode) or from the notebook menu.
#
# To change a text cell to a code cell, press `Y` in command mode or choose this option from the notebook menu.
#
#
# ### Keyboard Shortcuts
#
# If you intend to use Jupyter notebooks often, it is *strongly* recommend that you get used to using keyboard shortcuts, as it will greatly increase your productivity.
#
# Besides the ones given above, there are many others that could be useful, and can be found from a simple web search.

# %% [markdown]
# ### Entering Text
#
# Text cells are formatted with [Markdown](https://daringfireball.net/projects/markdown/), which provides a quick syntax to format text.  Click on [Syntax](https://daringfireball.net/projects/markdown/syntax) in the previous link to learn more about how to use it.
#
# Here are some of the basics:
#
# * **Italic:** `*italic*` produces *italic*.
# * **Bold:**  `**bold**` produces **bold**.
# * **Headers:**
#   - use `# Top Header Title` to produce a top level header;
#   - use `## Second Level Header Title` to produce a second level header;
#   - use `### Third Level Header Title` to produce a third level header;
# * **Links:** use `[link text](link URL)` to create links, for instance `[Python](https://www.python.org/)` produces [Python](https://www.python.org/).
#   
# Here is an example of an **bullet point list**:
#
# ```
# * first item
# * second item
# * third item
# ```
#
# produces
#
# * first item
# * second item
# * third item
#
#
# Here is an example of a **numbered list**:
#
# ```
# 1. first item
# 2. second item
# 3. third item
# ```
#
# produces
#
# 1. first item
# 2. second item
# 3. third item
#

# %% [markdown]
# ### Math in Text Cells
#
# You can also enter mathematical expressions using [LaTeX](https://www.latex-project.org/).  We will not go into much detail here, but there also countless resources for LaTeX online.
#
# But here are some of the basics:
#
# * Surround basic expressions with `$` to get **math formatted expressions**.  For instance, `$x + 1$` produces $x + 1$.  (Compare to what we get without the `$`'s: x + 1 versus $x + 1$.)
#
# * Use `^` for **powers** (between `$`'s), with braces `{ }` surrounding the powers.  For instance `$a^{2} + b^{2} = c^{2}$` produces $a^{2} + b^{2} = c^{2}$.
#
# * Use `\frac` or `\dfrac` to produce **fractions**, with braces `{ }` surrounding the numerator and denominator.  For instance:
#   - `$\frac{a^{2} + b^{2}}{c^{2}}$` produces $\frac{a^{2} + b^{2}}{c^{2}}$,
#   - `$\dfrac{a^{2} + b^{2}}{c^{2}}$` produces $\dfrac{a^{2} + b^{2}}{c^{2}}$.
#
# * Use `\sqrt` to produce **square roots**, with braces `{ }` surrounding what goes inside.  For instance:
#   - `$\sqrt{a^{2} + b^{2}}$` produces $\sqrt{a^{2} + b^{2}}$,
#   - `$\sqrt{\dfrac{a^{2} + b^{2}}{c^{2}}}$` produces $\sqrt{\dfrac{a^{2} + b^{2}}{c^{2}}}$.
#
# * Use `$\left({ ... }\right)$` to produce **adjustable parentheses**.  For instance `$\left({ \dfrac{a^{2} + {b^{2^{3}}}}{c^{2}} }\right)$` produces $\left({ \dfrac{a^{2} + {b^{2^{3}}}}{c^{2}} }\right)$.
#
#
#

# %% [markdown]
# ## Installation
#
# We will *not* go over the installation of [Python](https://www.python.org/) and [Jupyter Lab](https://jupyter.org/) here, but you can find instructions on their respective web sites.
#
# One easier way to install them, along with other Python packages for data science, that seems popular is to install [Anaconda](https://www.anaconda.com/).  On the other hand, the installation process takes a long time and the requires considerable amount of disk space.  Moreover, I've seen the process fail a couple of times for some of my students, so I feel somewhat reluctant to recommend it.
#
# Another alternative is to completely avoid installing it and using some online provider that has all you need already available.  Here are some options, although I only have personal experience with the first on the list:
#
# * [Cocalc](https://cocalc.com/)
# * [Google Colab](https://colab.research.google.com/)
# * [Kaggle](https://www.kaggle.com/)
#
# Cocalc is quite good and has a free tier, although I do pay a small monthly fee for it, so I am not sure how limited it is.  If they have not changed it since I last tried, it does work well, as long as you do not require a lot of computing power.  It seems good enough for experimentation with Python and Jupyter Notebooks.
#
# I've used Cocalc because it also provides other math tools (and I am a mathematician!), but the other two options seem popular for data science.

# %% [markdown]
# ## Numbers and Computations
#
# Python (within a Jupyter notebook or in its [shell](https://www.python.org/shell/), which allows us to enter single lines of code) can be used as a calculator.  For instance, to compute:
#
# $$
#  1 + \frac{2 \cdot 3}{4}
# $$
#
# we simply do:

# %%
1 + (2 * 3) / 4

# %% [markdown]
# (Again, press `Shift + Enter` to *evaluate* the cell and see the result.)
#
# The syntax for these computations are mostly intuitive, but it needs to be observed that Python uses `**` for exponentiation instead of the more usual `^`:

# %%
2 ** 3

# %% [markdown]
# Even worse, the symbol `^` does have a meaning, so you will not get an error when you run it, you will just get an unexpected (and incorrect, if you are expecting powers) result:

# %%
2 ^ 3

# %% [markdown]
# (If you are curious, the `^` is used as an [exclusive or](https://en.wikipedia.org/wiki/Exclusive_or), but it will not be important to us here.)
#
# As usual, one thing to be extremely careful when doing computations in a computer is the use of parentheses.  For example, if we want to compute $\dfrac{1}{2 + 3}$, we need `1 / (2 + 3)`, as `1 / 2 + 3` represents $\dfrac{1}{2} + 3$:

# %%
1 / (2 + 3)

# %%
1 / 2 + 3

# %% [markdown]
# Python has (basically) two types of numbers: `int` (for integers) and `float` (for "floating point").  In Python, they differ simply by introducing a decimal point.  Thus, we have that `2` and `2.0` *are not the same* in Python (as they are different kinds of numbers).
#
# One peculiarity of Python is that when using the division `/`, even an exact division of integers result in a float:

# %%
4 / 2

# %% [markdown]
# We can use `//` instead, which gives an integer:

# %%
4 // 2

# %% [markdown]
# But note that it gives the *quotient* of the division (as in long division) of integers:

# %%
14 // 4

# %% [markdown]
# The remainder of the division can be obtained with `%`:

# %%
14 % 4

# %% [markdown]
# Here are some of the most common operations:
#
# | Command | Operation |
# |---------|-----------|
# | `+` | Addition |
# | `-` | Subtraction |
# | `*` | Multiplication |
# | `/` | Division (with decimals) |
# | `**` | Exponentiation |
# | `//` | Quotient of division (for intergers) |
# | `%` | Remainder of division (for intergers) |

# %% [markdown]
# We also have a few math functions, such as `abs` for the *absolute value*:

# %%
abs(5)

# %%
abs(-5)

# %% [markdown]
# The function `round` rounds a float to a given number of decimals:

# %%
round(123.456789, 3)

# %% [markdown]
# ### The `math` Module
#
# Python does not have many math functions loaded from the start.  To get them we need to import the math module:

# %%
import math

# %% [markdown]
# After running the import command above, we get many mathematical functions (and some constants).  They all start with `math.`.  For instance, we can compute square roots with `math.sqrt`:

# %%
math.sqrt(16)

# %% [markdown]
# We also get `math.sin` to compute the sine:

# %%
math.sin(2)

# %% [markdown]
# We have `math.log` for the *natural log*:

# %%
math.log(5)

# %% [markdown]
# The number $\pi$ is obtained with `math.pi`:

# %%
math.pi  # no parentheses!

# %%
math.sin(math.pi / 2)

# %% [markdown]
# The base of the natural log (usually denoted by $e$) is given by `math.e`:

# %%
math.e

# %%
math.log(math.e)

# %% [markdown]
# We can find what the module `math` provides by running `help(math)`:

# %%
help(math)

# %% [markdown]
# ## Variables

# %% [markdown]
# We can store values in variables, so that these can be used later.  Here is an example of a computation of a restaurant bill:

# %%
subtotal = 30.17
tax_rate = 0.0925
tip_percentage = 0.2

tax = subtotal * tax_rate
tip = subtotal * tip_percentage

total = subtotal + tax + tip

round(total, 2)

# %% [markdown]
# Note how the variable names make clear what the code does, and allows us to reused it by changing the values of `subtotal`, `tax_rate`, and `tip_percentage`.

# %% [markdown]
# Variable names can only have:
# * letters (lower and upper case),
# * numbers, and
# * the underscore `_`.
#
# Moreover, variable names *cannot* start with a number and *should* not start with the underscore (unless you are aware of the [conventions for such variable names](https://peps.python.org/pep-0008/#naming-conventions)).
#
# You should always name your variables with descriptive names to make your code more readable.
#
# You should also try to avoid variable names already used in Python, as it would override their builtin values.  For instance, names like `print`, `int`, `abs`, `round` are already used in Python, so you should not used them.
#
# (If the name appears in a green in a code cell in Jupyter, then it is already taken!)

# %% [markdown]
# ## Comments
#
# We can enter *comments* in code cells to help describe what the code is doing.  Comments are text entered in Python (e.g., in code cells) that is ignored when running the code, so it is only present to provide information about the code.
#
# Comments in Python start with `#`.  All text after a `#` and in the same line is ignored by the Python interpreter.  (By convention, we usually leave two spaces between the code and `#` and one space after it.)
#
# As an illustration, here are some comments added to our previous restaurant code:

# %%
# compute restaurant bill

subtotal = 25.63  # meal cost in dollars
tax_rate = 0.0925  # tax rate
tip_percentage = 0.2  # percentage for the tip

tax = subtotal * tax_rate  # tax amount
tip = subtotal * tip_percentage  # tip amount

# compute the total:
total = subtotal + tax + tip

# round to two decimal places
round(total, 2)

# %% [markdown]
# Note that the code above probably did not need the comments, as it was already pretty clear.  Although there is such a thing as "too many comments", it is preferable to write too many than too few comments.

# %% [markdown]
# ## String (Text)
#
# *Strings* is the name for text blocks in Python (and in most programming languages).  To have a text (or string) object in Python, we simply surround it by single quotes `' '` or double quotes `" "`:

# %%
'This is some text.'

# %%
"This is also some text!"

# %% [markdown]
# If we need quotes inside the string, we need to use the other kind to delimit it:

# %%
"There's always time to learn something new."

# %%
'Descates said: "I think, therefore I am."'

# %% [markdown]
# What if we need both kinds of quotes in a string?
#
# We can *escape the quote* with a `\` as in:

# %%
"It's well know that Descartes has said: \"I think, therefore I am.\""

# %%
'It\'s well know that Descartes has said: "I think, therefore I am."'

# %% [markdown]
# Thus, when you repeat the string quote inside of it, put a `\` before it.
#
# Note that you can *always* escape the quotes, even when not necessary.  (It will do no harm.)  In the example below, there was no need to escape the single quote, as seen above:

# %%
"It\'s well know that Descartes has said: \"I think, therefore I am.\""

# %% [markdown]
# Another option is to use *triple quotes*, i.e., to surround the text by either `''' '''` or `""" """` (and then there is no need for escaping):

# %%
'''It's well know that Descartes has said: "I think, therefore I am."'''

# %% [markdown]
# On the other hand, we cannot use `""" """` here because our original string *ends* with a `"`.  If it did not, it would also work.  We can simply add a space:

# %%
"""It's well know that Descartes has said: "I think, therefore I am." """

# %% [markdown]
# Triple quote strings can also contain *multiple lines* (unlike single quote ones):

# %%
"""First line.
Second line.

Third line (after a blank line)."""

# %% [markdown]
# The output seems a bit strange (we have `\n` in place of line breaks --- we will talk about it below), but it *prints* correctly:

# %%
multi_line_text = """First line.
Second line.

Third line (after a blank line)."""

print(multi_line_text)

# %% [markdown]
# ### Special Characters
#
# The backslash `\` is used to give special characters.  (Note that it is *not* the forward slash `/` that is used for division!)
#
# Besides producing quotes (as in `\'` and `\"`), it can also produce line breaks, as seen above.
#
# For instance:

# %%
multi_line_text = "First line.\nSecond line.\n\nThird line (after a blank line)."

print(multi_line_text)

# %% [markdown]
# We can also use `\t` for *tabs*: it gives a "stretchable space" which can be convenient to align text:

# %%
aligned_text = "1\tA\n22\tBB\n333\tCCC\n4444\tDDDD"

print(aligned_text)

# %% [markdown]
# We could also use triple quotes to make it more readable:

# %%
aligned_text = """
1 \t A
22 \t BB
333 \t CCC
4444 \t DDDD"""

print(aligned_text)

# %% [markdown]
# Finally, if we need the backslash in our text, we use `\\` (i.e., we also *escape it*):

# %%
backslash_test = "The backslash \\ is used for special charaters in Python.\nTo use it in a string, we need double backslashes: \\\\."

print(backslash_test)

# %% [markdown]
# ### f-Strings
#
# [f-strings](https://docs.python.org/3/reference/lexical_analysis.html#f-strings) (or *formatted string literals*) are helpful when you want to print variables with a string.
#
# For example:

# %%
birth_year = 2008
current_year = 2023

print(f"I was born in {birth_year}, so I am {current_year - birth_year} years old.")

# %% [markdown]
# So, we need to preface our (single quoted or double quoted) string with `f` and put our expression inside curly braces `{ }`.  It can be a variable (as in `birth_year`) or an expression.
#
# f-strings also allow us to format the expressions inside braces.  (Check the [documentation](https://docs.python.org/3/reference/lexical_analysis.html#f-strings) if you want to learn more.)

# %% [markdown]
# ### String Manipulation
#
# We can concatenate string with `+`:

# %%
name = "Alice"
eye_color = "brown"

name + " has " + eye_color + " eyes."

# %% [markdown]
# Note that we could have use an f-sting in the example above:

# %%
f"{name} has {eye_color} eyes."

# %% [markdown]
# We also have *methods* to help us manipulate strings.
#
# *Methods* are functions that belong to a particular object type, like strings, integers, and floats.  The syntax is `object.method(arguments)`.
#
# We can convert to upper case with the `upper` method:

# %%
test_string = "abc XYZ 123"
test_string.upper()

# %% [markdown]
# Similarly, the method `lower` converts to lower case:

# %%
test_string.lower()

# %% [markdown]
# We can also spit a string into a *list* of strings (more about lists below) with `split`:

# %%
test_string.split()

# %% [markdown]
# By default, it splits on spaces, but you can give a different character as an argument to specify the separator:

# %%
"abc-XYZ-123".split("-")

# %%
"abaccaaddd".split("a")

# %% [markdown]
# ## Lists
#
# *Lists* are (ordered) sequences of Python objects.  To create at list, you surround the elements by square brackets `[ ]` and separate them with commas `,`.  For example:

# %%
list_of_numbers = [5, 7, 3, 2]

list_of_numbers

# %% [markdown]
# But lists can have elements of any type:

# %%
mixed_list = [0, 1.2, "some string", [1, 2, 3]]

mixed_list

# %% [markdown]
# We can also have an empty list (to which we can later add elements):

# %%
empty_list = []

empty_list

# %% [markdown]
# ### Ranges

# %% [markdown]
# We can also create lists of consecutive numbers using `range`.  For instance, to have a list with elements from 0 to 5 we do:

# %%
list(range(6))

# %% [markdown]
# (Technically, `range` gives an object similar to a list, but not quite the same.  Using the function `list` we convert this object to an actual list.  Most often we do *not* need to convert the object to a list in practice, though.)
#
# Note then that `list(range(n))` gives a list `[0, 1, 2, ..., n - 1]`, so `n` itself is *not* included!  (This is huge pitfall when first learning with Python!)
#
# We can also tell where to start the list (if not at 0), by passing two arguments:

# %%
list(range(3, 10))

# %% [markdown]
# In this case the list start at 3, but ends at 9 (and not 10).
#
# We can also pass a third argument, which is the *step size*:

# %%
list(range(4, 20, 3))

# %% [markdown]
# So, we start at exactly the first argument (4 in this case), skip by the third argument (3 in this case), and stop in the last number *before* the second argument (20 in this case).

# %% [markdown]
# ### Extracting Elements
#
# First, remember our `list_of_numbers` and `mixed_list`:

# %%
list_of_numbers

# %%
mixed_list

# %% [markdown]
# We can extract elements from a list by position.  But, **Python counts from 0** and not 1.  So, to extract the first element of `list_of_numbers` we do:

# %%
list_of_numbers[0]

# %% [markdown]
# To extract the second:

# %%
list_of_numbers[1]

# %% [markdown]
# We can also count from the end using *negative indices*.  So, to extract the last element we use index `-1`:

# %%
mixed_list[-1]

# %% [markdown]
# The element before last:

# %%
mixed_list[-2]

# %% [markdown]
# To extract the `2` from `[1, 2, 3]` in `mixed_list`:

# %%
mixed_list[3][1]

# %% [markdown]
# (`[1, 2, 3]` is at index `3` of `mixed_list`, and `2` is at index `1` of `[1, 2, 3]`.)

# %% [markdown]
# ### Slicing
#
# We can get sublists from a list using what is called *slicing*.  For instance, let's start with the list:

# %%
list_example = list(range(5, 40, 4))

list_example

# %% [markdown]
# If I want to get a sublist of `list_example` starting at index 3 and ending at index 6, we do:

# %%
list_example[3:7]

# %% [markdown]
# **Note we used 7 instead of 6!**  Just like with ranges, we stop *before* the second number.
#
# If we want to start at the beginning, we can use 0 for the first number, or simply omit it altogether:

# %%
list_example[0:5]  # first 5 elements -- does not include index 5

# %%
list_example[:5]  # same as above

# %% [markdown]
# Omitting the second number, we go all the way to the end:

# %%
list_example[-3:]

# %% [markdown]
# We can get the length of a list with the function `len`:

# %%
len(list_example)

# %% [markdown]
# So, we could also do:

# %%
list_example[4:len(list_example)]  # all elements from index 4 until the end

# %% [markdown]
# Note that the last valid index of the list is `len(list_example) - 1`, and *not* `len(list_example)`, since, again, we start counting from 0 and not 1.

# %% [markdown]
# We can also give a step size for the third argument, similar to `range`:

# %%
new_list = list(range(31))

new_list

# %%
new_list[4:25:3]

# %% [markdown]
# ### Changing a List
#
# We can also *change elements* in a list.
#
# First, recall our `list_of_numbers`:

# %%
list_of_numbers

# %% [markdown]
# If then, for instance, we want to change the element at index 2 in `list_of_numbers` (originally a 3) to a 10, we can do:

# %%
list_of_numbers[2] = 10

list_of_numbers

# %% [markdown]
# We can add an element to the end of a list using the `append` *method*.  So, to add $-1$ to the end of `list_of_numbers`, we can do:

# %%
list_of_numbers.append(-1)

list_of_numbers

# %% [markdown]
# Note that `append` *changes the original list* and *returns no output*!

# %% [markdown]
# We can sort with the `sort` method:

# %%
list_of_numbers.sort()

list_of_numbers

# %% [markdown]
# (Again, it *changes the list and returns no output*!)

# %% [markdown]
# To sort in reverse order, we can use the optional argument `reverse=True`:

# %%
list_of_numbers.sort(reverse=True)

list_of_numbers

# %% [markdown]
# We can reverse the order of elements with the `reverse` method.  (This method does *no sorting at all*, it just reverse the whole list in its given order.)

# %%
mixed_list

# %%
mixed_list.reverse()

mixed_list

# %% [markdown]
# We can remove elements with the `pop` method.  By default it removes the last element of the list, but you can also pass it the index of the element to removed.
#
# `pop` changes the original list *and* returns the element removed!

# %%
list_of_numbers

# %%
removed_element = list_of_numbers.pop()  # remove last element

removed_element

# %%
list_of_numbers  # the list was changed!

# %%
removed_element = list_of_numbers.pop(1)  # remove element at index 1

removed_element

# %%
list_of_numbers  # again, list has changed!

# %% [markdown]
# ### List and Strings
#
# One can think of strings as (more or less) lists of characters.  (This is not 100% accurate, as we will see, but it is pretty close.)
#
# So, many of the operations we can do with list, we can also do with strings.
#
# For instance, we can use `len` to find the lenght (or number of characters) of a string:

# %%
quote = "I think, therefore I am."

len(quote)

# %% [markdown]
# We can also extract elements by index:

# %%
quote[3]  # 4th character

# %% [markdown]
# And, we can slice a string:

# %%
quote[2:20:3]

# %% [markdown]
# Conversely, just as we could concatenate strings with `+`, we can concatenate lists with `+`:

# %%
[1, 2, 3] + [4, 5, 6, 7]

# %% [markdown]
# The crucial difference is that **we cannot change a string** (like we can change a list).
#
# If, for instance, you try
#
# ```python
# quote[3] = "X"
# ```
#
# you get an error.

# %% [markdown]
# Finally, if we have a list of strings, we can join them with the *string* method`join`.  (It is not a *list* method.)  The string in question is used to *separate* the strings in the list.  For instance:

# %%
list_of_strings = ["all", "you", "need", "is", "love"]

" ".join(list_of_strings)

# %%
"---".join(list_of_strings)

# %%
"".join(list_of_strings)

# %% [markdown]
# ## Dictionaries
#
# *Dictionaries* are used to store data that can be retrieve from a *key*, instead of from position.  (In principle, a dictionary has no order!)  So, to each *key* (which must be unique) we have an associate *value*.
#
# You can think of a real dictionary, where you look up definitions for a word.  In this example the keys are the words, and the values are the definitions.
#
# In Python's dictionaries we have the key/value pairs surrounded by curly braces `{ }` and separated by commas `,`, and the key/value pairs are separated by a colon `:`.
#
# For instance, here is a dictionary with the weekdays in French:

# %%
french_days = {"Sunday": "dimanche", "Monday": "lundi", "Tuesday": "mardi", 
               "Wednesday": "mercredi", "Thursday": "jeudi", "Friday": "vendredi", "Saturday": "samedi"}

french_days

# %% [markdown]
# (Here the keys are the days in English, and to each key the associate value is the corresponding day in French.)
#
#
# Then, when I want to look up what is Thursday in French, I can do:

# %%
french_days["Thursday"]

# %% [markdown]
# As another example, we can have a dictionary that has all the grades (in a list) o students in a course:

# %%
grades = {"Alice": [89, 100, 93], "Bob": [78, 83, 80], "Carl": [85, 92, 100]}

grades

# %% [markdown]
# To see Bob's grades:

# %%
grades["Bob"]

# %% [markdown]
# To get the grade of Carl's second exam:

# %%
grades["Carl"][1]

# %% [markdown]
# We can also add a pair of key/value to a dictionary.  For instance, to enter Denise's grades, we can do:

# %%
grades["Denise"] = [98, 93, 100]

grades

# %% [markdown]
# We can also change the values:

# %%
grades["Bob"] = [80, 85, 77]

grades

# %% [markdown]
# Or, to change a single grade:

# %%
grades["Alice"][2] = 95

grades

# %% [markdown]
# We can use `pop` to remove a pair of key/value by passing the corresponding key.  It returns the *value* for the given key and changes the dictionary (by removing the pair):

# %%
bobs_grades = grades.pop("Bob")

bobs_grades

# %%
grades

# %% [markdown]
# ## Conditionals
#
# ### Booleans
#
# Python has two reserved names for true and false: `True` and `False`.  (Note it *must* be capitalized for Python to recognize them as booleans!  `true` and `false` do not work!)
#
# For instance:

# %%
2 < 3

# %%
2 > 3

# %% [markdown]
# One can flip their values with `not`:

# %%
not (2 < 3)

# %%
not (3 < 2)

# %%
not True

# %%
not False

# %% [markdown]
# These can also be combined with `and` and `or`:

# %%
(2 < 3) and (4 < 5)

# %%
(2 < 3) and (4 > 5)

# %%
(2 < 3) or (4 > 5)

# %%
(2 > 3) or (4 > 5)

# %% [markdown]
# Note that `or` is not exclusive (as usually in common language).  In a restaurant, if an entree comes with "soup or salad", both is *not* an option.  But in math and computer science, `or` allows both possibilities being true:

# %%
(2 < 3) or (4 < 5)

# %% [markdown]
# ### Comparisons
#
# We have the following comparison operators:
#
# | **Operator** | **Description** |
# |--------------|-----------------|
# | `==`         | Equality ($=$)  |
# | `!=`         | Different ($\neq$) |
# | `<`          | Less than ($<$) |
# | `<=`         | Less than or equal to ($\leq$) |
# | `>`          | Greater than ($>$) |
# | `>=`         | Greater than or equal to ($\geq$) |
#
#
# Note that since we use `=` to assign values to variables, we need `==` for comparisons.  
#
# *It's a common mistake to try to use `=` in a comparison, so be careful!*

# %% [markdown]
# Note that we can use
#
# ```python
# 2 < 3 <= 4
# ```
#
# as a shortcut for
#
# ```python
# (2 < 3) and (3 <= 4)
# ```

# %%
2 < 3 <= 4

# %%
2 < 5 <= 4

# %% [markdown]
# #### String Comparisons
#
# Note that these can also be used with other objects, such as strings:

# %%
"alice" == "alice"

# %%
"alice" == "bob"

# %% [markdown]
# It's case sensitive:

# %%
"alice" == "Alice"

# %% [markdown]
# The inequalities follow *dictionary order*:

# %%
"aardvark" < "zebra"

# %%
"giraffe" < "elephant"

# %%
"car" < "care"

# %% [markdown]
# But note that capital letters come earlier than all lower case letters:

# %%
"Z" < "a"

# %%
"aardvark" < "Zebra"

# %% [markdown]
# ### Methods that Return Booleans
#
# We have functions/methods that return booleans.
#
# For instance, to test if a string is made of lower case letters:

# %%
test_string = "abc"

test_string.islower()

# %%
test_string = "aBc"

test_string.islower()

# %%
test_string = "abc1"

test_string.islower()

# %% [markdown]
# Here some other methods for strings:
#
# | **Method** | **Description** |
# |------------|-----------------|
# | `is.lower` | Checks if all letters are lower case |
# | `is.upper` | Checks if all letters are upper case |
# | `is.alnum` | Checks if all characters are letters and numbers |
# | `is.alpha` | Checks if all characters are letters |
# | `is.numeric` | Checks if all characters are numbers |
#

# %% [markdown]
# ### Membership
#
# We can test for membership with the keywords `in`:

# %%
2 in [1, 2, 3]

# %%
5 in [1, 2, 3]

# %%
1 in [0, [1, 2, 3], 4]

# %%
[1, 2, 3] in [0, [1, 2, 3], 4]

# %% [markdown]
# It also work for strings:

# %%
"vi" in "evil"

# %%
"vim" in "evil"

# %% [markdown]
# Note the the character must appear together:

# %%
"abc" in "axbxc"

# %% [markdown]
# We can also write `not in`.  So
#
# ```python
# "vim" not in "evil"
# ```
#
# is the same as 
#
# ```python
# not "vim" in "evil"
# ```

# %%
"vim" not in "evil"

# %% [markdown]
# ## if-Statements
#
# We can use conditionals to decide what code to run using *if-statements*:

# %%
water_temp = 110  # in Celsius

if water_temp >= 100:
    print("Water will boil.")

# %%
water_temp = 80  # in Celsius

if water_temp >= 100:
    print("Water will boil.")

# %% [markdown]
# The syntax is:
#
# ```
# if <condition>:
#     <code to run if condition is true>
# ```

# %% [markdown]
# Note the indentation: all code that is indented will run when the condition is true!

# %%
water_temp = 110  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
    print("(Temperature above 100.)")

# %%
water_temp = 80  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
print("Non-indented code does not depend on the condition!")

# %% [markdown]
# We can add an `else` statement for code we want to run *only when the condition is false*:

# %%
water_temp = 110  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
else:
    print("Water will not boil.")

print("This will always be printed.")

# %%
water_temp = 80  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
else:
    print("Water will not boil.")

print("This will always be printed.")

# %% [markdown]
# We can add more conditions with `elif`, which stands for *else if*.  
#
# For instance, if we want to check if the water will freeze:

# %%
water_temp = 110  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
elif water_temp <= 0:
    print("Water will freeze.")

# %%
water_temp = -5  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
elif water_temp <= 0:
    print("Water will freeze.")

# %%
water_temp = 50  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
elif water_temp <= 0:
    print("Water will freeze.")

# %% [markdown]
# Note that if we have overlapping conditions, only the *first* to be met runs!

# %%
number = 70

if number > 50:
    print("First condition met.")
elif number > 30:
    print("Second condition met, but not first")

# %%
number = 40

if number > 50:
    print("First condition met.")
elif number > 30:
    print("Second condition met, but not first")

# %%
number = 20

if number > 50:
    print("First condition met.")
elif number > 30:
    print("Second condition met, but not first")

# %% [markdown]
# We can add an `else` at the end, which will run when all conditions above it (from `if` an `elif`'s) are false:

# %%
water_temp = 110  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
elif water_temp <= 0:
    print("Water will freeze.")
else:
    print("Water will neither boil, nor freeze.")

# %%
water_temp = -5  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
elif water_temp <= 0:
    print("Water will freeze.")
else:
    print("Water will neither boil, nor freeze.")

# %%
water_temp = 40  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
elif water_temp <= 0:
    print("Water will freeze.")
else:
    print("Water will neither boil, nor freeze.")

# %% [markdown]
# We can have as many `elif`'s as we need:

# %%
water_temp = 110  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
elif water_temp >= 90:
    print("Water is close to boiling!")
elif 0 < water_temp <= 10:
    print("Water is close to freezing!")
elif water_temp <= 0:
    print("Water will freeze.")
else:
    print("Water will neither boil, nor freeze, nor it is close to either.")

# %%
water_temp = 90  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
elif water_temp >= 90:
    print("Water is close to boiling!")
elif 0 < water_temp <= 10:
    print("Water is close to freezing!")
elif water_temp <= 0:
    print("Water will freeze.")
else:
    print("Water will neither boil, nor freeze, nor it is close to either.")

# %%
water_temp = 40  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
elif water_temp >= 90:
    print("Water is close to boiling!")
elif 0 < water_temp <= 10:
    print("Water is close to freezing!")
elif water_temp <= 0:
    print("Water will freeze.")
else:
    print("Water will neither boil, nor freeze, nor it is close to either.")

# %%
water_temp = 3  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
elif water_temp >= 90:
    print("Water is close to boiling!")
elif 0 < water_temp <= 10:
    print("Water is close to freezing!")
elif water_temp <= 0:
    print("Water will freeze.")
else:
    print("Water will neither boil, nor freeze, nor it is close to either.")

# %%
water_temp = -5  # in Celsius

if water_temp >= 100:
    print("Water will boil.")
elif water_temp >= 90:
    print("Water is close to boiling!")
elif 0 < water_temp <= 10:
    print("Water is close to freezing!")
elif water_temp <= 0:
    print("Water will freeze.")
else:
    print("Water will neither boil, nor freeze, nor it is close to either.")

# %% [markdown]
# Note that we could also have used instead
#
# ```python
# if water_temp >= 100:
#     print("Water will boil.")
# elif water_temp >= 90:
#     print("Water is close to boiling!")
# elif water_temp <= 0:
#     print("Water will freeze.")
# elif water_temp <= 10:
#     print("Water is close to freezing!")
# else:
#     print("Water will neither boil, nor freeze, nor it is close to either.")
# ```
#
# but *not*
#
# ```python
# if water_temp >= 100:
#     print("Water will boil.")
# elif water_temp >= 90:
#     print("Water is close to boiling!")
# elif water_temp <= 10:
#     print("Water is close to freezing!")
# elif water_temp <= 0:
#     print("Water will freeze.")
# else:
#     print("Water will neither boil, nor freeze, nor it is close to either.")
# ```

# %%
water_temp = -5  # should say it is freezing!

if water_temp >= 100:
    print("Water will boil.")
elif water_temp >= 90:
    print("Water is close to boiling!")
elif water_temp <= 10:
    print("Water is close to freezing!")
elif water_temp <=0:
    print("Water will freeze.")
else:
    print("Water will neither boil, nor freeze, nor it is close to either.")

# %% [markdown]
# ## for Loops
#
# We can use *for-loops* for repeating tasks.
#
# Let's show its use with an example.
#
# ### Loops with `range`
#
# To print *Beetlejuice* three times we can do:

# %%
for i in range(3):
    print("Beetlejuice")

# %% [markdown]
# The `3` in `range(3)` is the number of repetitions, and the indented block below the `for` line is the code to be repeated.  The `i` is the *loop variable*, but it is not used in this example.  (We will examples when we do use it soon, though.)
#
# Here `range(3)` can be thought as the list `[0, 1, 2]` (as seen above), and in each of the three times that the loop runs, the loop variable, `i` in this case, receives one of the values in this list *in order*.
#
# Let's illustrate this with another example:

# %%
for i in range(3):
    print(f"The value of i is {i}")  # print the value of i

# %% [markdown]
# So, the code above is equivalent to running:

# %%
# first iteration
i = 0
print(f"The value of i is {i}")

# second iteration
i = 1
print(f"The value of i is {i}")

# third iteration
i = 2
print(f"The value of i is {i}")

# %% [markdown]
# Here the `range` function becomes quite useful (and we should not surround it by `list`!).  For instance, if we want to add all even numbers, between 4 and 200 (both inclusive), we could do:

# %%
total = 0  # start with 0 as total

for i in range(2, 201, 2):  # note the 201 instead of 200!
    total = total + i  # replace total by its current value plus the value of i

print(total)  # print the result

# %% [markdown]
# It's worth observing that `total += i` is a shortcut (and more efficient than) `total = total + i`, so we could have done:

# %%
total = 0  # start with 0 as total

for i in range(2, 201, 2):  # note the 201 instead of 200!
    total += i  # replace total by its current value plus the value of i

print(total)  # print the result

# %% [markdown]
# Let's now create a list with the first $10$ perfect squares:

# %%
squares = []  # start with an empty list

for i in range(10):  # i = 0, 1, 2, ... 9
    squares.append(i ** 2)  # add i ** 2 to the end of squares

squares

# %% [markdown]
# ### Loops with Lists
#
# One can use any list instead of just `range`.  For instance:

# %%
languages = ["Python", "Java", "C", "Rust", "Julia"]

for language in languages:
    print(f"{language} is a programming language.")

# %% [markdown]
# The code above is equivalent to

# %%
language = "Python"
print(f"{language} is a programming language.")

language = "Java"
print(f"{language} is a programming language.")

language = "C"
print(f"{language} is a programming language.")

language = "Rust"
print(f"{language} is a programming language.")

language = "Julia"
print(f"{language} is a programming language.")

# %% [markdown]
# ### Loops with Dictionaries
#
# We can also loop over dictionaries.  In this case the loop variable receives the *keys* of the dictionary:

# %%
french_days

# %%
for day in french_days:
    print(f"{day} in French is {french_days[day]}.")

# %% [markdown]
# ### List Comprehensions
#
# Python has a shortcut to create lists that we would usually created with a for loop.  It is easier to see how it works with a couple of examples.
#
# Suppose we want to create a function with the first ten positive cubes.  We can start with an empty list and add the cubes in a loop, as so:

# %%
# empty list
cubes = []

for i in range(1, 11):
    cubes.append(i ** 3)

cubes

# %% [markdown]
# Using *list comprehension*, we can obtain the same list with:

# %%
cubes = [i ** 3 for i in range(1, 11)]

cubes

# %% [markdown]
# Here is a more complex example.  Suppose we want to create a list of lists like:
#
# ```python
# [[1],
#  [1, 2],
#  [1, 2, 3], 
#  [1, 2, 3, 4],
#  [1, 2, 3, 4, 5]]
# ```
#
# To do that, we need *nested for loops:

# %%
nested_lists = []

for i in range(1, 6):
    inner_list = []
    for j in range(1, i + 1):
        inner_list.append(j)
    nested_lists.append(inner_list)

nested_lists

# %% [markdown]
# (Note that we could have replaced the inner loop with `inner_list = list(range(1, i + 1)`, but let's keep the loops to illustrate the mechanics of the process of changing from loops to list comprehensions.)
#
# Here is how we can do it using list comprehension:

# %%
nested_lists = [[j for j in range(1, i + 1)] for i in range(1, 6)]

nested_lists


# %% [markdown]
# ## Functions
#
# You are probably familiar with functions in mathematics.  For instance, if $f(x) = x^2$, then $f$ take some number $x$ as its *input* and returns its square $x^2$ as the *output*.  So,
#
# $$
# \begin{align*}
#   f(1) &= 1^2 = 1, && \text{(input: $1$, output: $1$)}; \\
#   f(2) &= 2^2 = 4, && \text{(input: $2$, output: $4$)}; \\
#   f(3) &= 3^2 = 9, && \text{(input: $3$, output: $9$)}; \\
#   f(4) &= 4^2 = 16, && \text{(input: $4$, output: $16$)}.
# \end{align*}
# $$

# %% [markdown]
# We can do the same in Python:

# %%
def square(x):
    return x ** 2


# %% [markdown]
# Here is a brief description of the syntax:
#
# * `def` is the keyword that tell Python we are *defining* a function;
# * `square` is the name of the function we chose (it has the same requirements as variable names);
# * inside the parentheses after the name come the parameter(s), i.e., the inputs of the function, in this case only `x`;
# * indented comes the code that runs when the function is called;
# * `return` gives the value that will be returned by the function, i.e., the output.
#
# Now to run, we just use the name with the desired input inside the parentheses:

# %%
square(1)

# %%
square(2)

# %%
square(3)

# %%
square(4)


# %% [markdown]
# It is *strongly recommended* that you add a *docstring* describing the function right below its `def` line.  We use triple quotes for that:

# %%
def square(x):
    """
    Given a value x, returns its square x ** 2.

    INPUT:
    x: a number.

    OUTPUT:
    The square of the input.
    """
    return x ** 2


# %% [markdown]
# It does not affect how the function works:

# %%
square(3)

# %% [markdown]
# But it allows whoever reads the code for the function to understand what it does.  (This might be *you* after a few days not working on the code!)
#
# It also allows anyone to get help for the function:

# %%
help(square)

# %% [markdown]
# Functions are like mini-programs.  For instance, remember the code to compute a restaurant bill:

# %%
# compute restaurant bill

subtotal = 25.63  # meal cost in dollars
tax_rate = 0.0925  # tax rate
tip_percentage = 0.2  # percentage for the tip

tax = subtotal * tax_rate  # tax amount
tip = subtotal * tip_percentage  # tip amount

# compute the total:
total = subtotal + tax + tip

# round to two decimal places
round(total, 2)


# %% [markdown]
# We can turn it into a function!  We can pass `subtotal`, `tax_rate`, and `tip_percentage` as arguments, and get the total.
#
# Here is how it is done:

# %%
def restaurant_bill(subtotal, tax_rate, tip_percentage):
    """
    Given the subtotal of a meal, tax rate, and tip percentage, returns
    the total for the bill.

    INPUTS:
    subtotal: total cost of the meal (before tips and taxes);
    tax_rate: the tax rate to be used;
    tip_percentage: percentage of subtotal to be used for the tip.

    OUTPUT:
    Total price of the meal with taxes and tip.
    """
    tax = subtotal * tax_rate  # tax amount
    tip = subtotal * tip_percentage  # tip amount

    # compute the total:
    total = subtotal + tax + tip

    # return total rounded to two decimal places
    return round(total, 2)


# %% [markdown]
# So, `restaurant_bill(25.63, 0.0925, 0.2)` should return the same value as above, `33.13`:

# %%
restaurant_bill(25.63, 0.0925, 0.2)

# %% [markdown]
# But now we can use other values, without having to type all the code again.  For instance, if the boll was $\$30$, tax rate is $8.75\%$, and we tip $18\%$, our bill comes to:

# %%
restaurant_bill(30, 0.0875, 0.18)


# %% [markdown]
# ### Default Values
#
# If we the tax rate and tip percentages don't usually change, we can set some default values for them in our function.  
#
# For instance, let's assume that the tax rate is usually $9.25\%$ and the tip percentage is $20\%$.  We just set these values in the declaration of the function.  I also change the docstring to reflect the changes, but the rest remains the same.

# %%
def restaurant_bill(subtotal, tax_rate=0.0925, tip_percentage=0.2):
    """
    Given the subtotal of a meal, tax rate, and tip percentage, returns
    the total for the bill.

    INPUTS:
    subtotal: total cost of the meal (before tips and taxes);
    tax_rate: the tax rate to be used;
              default value: 0.0925 (9.25%);
    tip_percentage: percentage of subtotal to be used for the tip;
                    default value: 0.2 (20%).

    OUTPUT:
    Total price of the meal with taxes and tip.
    """
    tax = subtotal * tax_rate  # tax amount
    tip = subtotal * tip_percentage  # tip amount

    # compute the total:
    total = subtotal + tax + tip

    # return total rounded to two decimal places
    return round(total, 2)


# %% [markdown]
# Now, every time I use the default values, we can omit them:

# %%
restaurant_bill(25.63)

# %% [markdown]
# But I still can change them!  If I want to give a tip of $22\%$, I can do:

# %%
restaurant_bill(25.63, tip_percentage=0.22)

# %% [markdown]
# And if I am at a different state, where the tax rate is $8.75\%$:

# %%
restaurant_bill(25.63, tax_rate=0.0875)

# %% [markdown]
# And I can alter both, of course:

# %%
restaurant_bill(30, tax_rate=0.0875, tip_percentage=0.18)

# %% [markdown]
# ### Lambda (or Nameless) Functions
#
# We can create simple one line functions with a shortcut, using the `lambda` keyword.
#
# For instance, here is how we can create the `square` function from above with:

# %%
square = lambda x: x ** 2

# %% [markdown]
# Here is a description of the syntax:
#
# * `square =` just tells to store the result of the expression following `=` into the variable `square` (as usual).  In this case, the expression gives a *function*.
# * `lambda` is the keyword that tells Python we are creating a (lambda) function.
# * What comes before the `:` are the arguments of the function (only `x` in this case).
# * What comes after the `:` is what the function returns (`x ** 2` in this case).  (It must be a single line, containing what would come after `return` in a regular function.)
#
# Again, except for the docstring, which we *cannot* add with lambda functions, the code is equivalent to what we had before for the `square` function.

# %%
square(3)

# %%
square(4)

# %% [markdown]
# Here is another example, with two arguments:

# %%
average_of_two = lambda x, y: (x + y) / 2

# %%
average_of_two(3, 7)

# %%
average_of_two(5, 6)


# %% [markdown]
# **Note:** The most common use for lambda functions is to create functions that we pass *as arguments to other functions or methods*.  
#
# In this scenario, we do not need to first create a function with `def`, giving it a name, and then pass this name as the argument of the other function/method.  We can simply create the function *inside the parentheses of the argument of the function*.  Thus, we do not need to name this function in the argument, which is why we sometimes call these lambda functions *nameless*.
#
# Here is an example.  Let's create a function that takes another function as an argument and returns the result of this function when evaluated at $1$:

# %%
def evaluate_at_1(function):
    """
    Evaluates given function at 1.

    INPUT:
    function: some funciton with one (numerical) argument.

    OUTPUT:
    The function evaluated at 1.

    """
    return function(1)


# %% [markdown]
# So, if we call `evaluate_at_1(square)`, we should get `1 ** 2`, i.e., `1`:

# %%
evaluate_at_1(square)


# %% [markdown]
# Now consider the function `add_one`:

# %%
def add_one(x):
    """
    Given x, returns x + 1.

    INPUT:
    x: some number.

    OUTPUT:
    The number given plus 1.
    """
    return x + 1


# %% [markdown]
# Now, if we call `evaluate_at_one(add_one)`, we should get `1 + 1`, i.e., `2`:

# %%
evaluate_at_1(add_one)

# %% [markdown]
# I could also create a function `add_two` that would return the input plus $2$.  But if all I need of this function is to pass it as an argument to `evaluate_at_1`, I can create it directly, without first creating and giving it a name:

# %%
evaluate_at_1(lambda x: x + 2)

# %% [markdown]
# This example is a bit artificial, but this need to pass functions as arguments often occurs in practice, and lambda functions come handy.

# %% [markdown]
# ## Comments, Suggestions, Corrections
#
# Please send your comments, suggestions, and corrections to lfinotti@utk.edu.
