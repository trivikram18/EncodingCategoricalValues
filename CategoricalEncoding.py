# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:41:56 2019

@author: trivikram.cheedella

@source: http://pbpython.com/categorical-encoding.html
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# path = 'C:/Users/trivi/Google Drive/DataScience/Kaggle/Python/Pandas'
path = 'C:/Users/trivikram.cheedella/OneDrive - JD Power/Google Drive/DataScience/EncodingCategoricalValues'
os.chdir(path)
print (os.getcwd())

# Define the headers since the data does not have any
headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]

# Read in the CSV file and convert "?" to NaN
df = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/autos/imports-85.data",
                  header=None, names=headers, na_values="?" )
df.head()

# Check the data types
df.dtypes

# Separate out the object type variables as we are only focusing on encoding the categorical values
obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()

# Check for null values
obj_df[obj_df.isnull().any(axis=1)]

obj_df["num_doors"].value_counts()

# For the sake of simplicity, just fill in the value with the number 4 (since that is the most common value):
obj_df = obj_df.fillna({"num_doors": "four"})

### Approach #1 - Find and Replace ###
obj_df["num_cylinders"].value_counts()

# cleaning up the num_doors and num_cylinders columns:
cleanup_nums = {"num_doors":     {"four": 4, "two": 2},
                "num_cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}

obj_df.replace(cleanup_nums, inplace=True)
obj_df.head()

# The nice benefit to this approach is that pandas “knows” the types of values in the columns 
# so the object is now a int64
obj_df.dtypes

### Approach #2 - Label Encoding ###
"""
For example, the body_style column contains 5 different values. We could choose to encode it like this:
convertible -> 0
hardtop -> 1
hatchback -> 2
sedan -> 3
wagon -> 4
"""

# One trick you can use in pandas is to convert a column to a category, then use those category values for your label encoding:
obj_df["body_style"] = obj_df["body_style"].astype('category')
obj_df.dtypes
obj_df["body_style"].value_counts()

# Then you can assign the encoded variable to a new column using the cat.codes accessor:
obj_df["body_style_cat"] = obj_df["body_style"].cat.codes
obj_df.head()
obj_df.dtypes

# The nice aspect of this approach is that you get the benefits of pandas categories 
# (compact data size, ability to order, plotting support) 
# but can easily be converted to numeric values for further analysis.

##### Approach #3 - One Hot Encoding #####
"""
Label encoding has the advantage that it is straightforward but it has the disadvantage that the numeric values 
can be “misinterpreted” by the algorithms. 
For example, the value of 0 is obviously less than the value of 4 but does that really correspond to the 
data set in real life? Does a wagon have “4X” more weight in our calculation than the convertible? 
In this example, I don’t think so.
"""
pd.get_dummies(obj_df, columns=["drive_wheels"]).head()

# Proper naming will make the rest of the analysis just a little bit easier.
pd.get_dummies(obj_df, columns=["body_style", "drive_wheels"], prefix=["body", "drive"]).head()

# The other concept to keep in mind is that get_dummies returns the full dataframe 
# so you will need to filter out the objects using select_dtypes when you are ready to do the final analysis.

##### Approach #4 - Custom Binary Encoding #####
"""
Depending on the data set, you may be able to use some combination of label encoding and one hot encoding to 
create a binary column that meets your needs for further analysis.
"""

obj_df["engine_type"].value_counts()
obj_df["OHC_Code"] = np.where(obj_df["engine_type"].str.contains("ohc"), 1, 0)
obj_df[["make", "engine_type", "OHC_Code"]].head(10)


##### Scikit-Learn #####
"""
In addition to the pandas approach, scikit-learn provides similar functionality. 
Using pandas a little simpler but it is important to be aware of how to execute the processes in scikit-learn.

For instance, if we want to do a label encoding on the make of the car, 
we need to instantiate a LabelEncoder object and fit_transform the data:
"""
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
obj_df["make_code"] = lb_make.fit_transform(obj_df["make"])
obj_df[["make", "make_code"]].head(11)

"""
Scikit-learn also supports binary encoding by using the LabelBinarizer. 
We use a similar process as above to transform the data but the process of creating a 
pandas DataFrame adds a couple of extra steps.
"""
from sklearn.preprocessing import LabelBinarizer

lb_style = LabelBinarizer()
lb_results = lb_style.fit_transform(obj_df["body_style"])
pd.DataFrame(lb_results, columns=lb_style.classes_).head()

"""
The next step would be to join this data back to the original dataframe
"""

##### Advanced Approaches #####
"""
There are even more advanced algorithms for categorical encoding. 
http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/
The other nice aspect is that the author of the article has created a scikit-learn contrib package 
called categorical-encoding
http://contrib.scikit-learn.org/categorical-encoding/
"""

"""
Here is a brief introduction to using the library for some other types of encoding. 
For the first example, we will try doing a Backward Difference encoding.
First we get a clean dataframe and setup the BackwardDifferenceEncoder :
"""
import category_encoders as ce

# Get a new clean dataframe
obj_df = df.select_dtypes(include=['object']).copy()

# Specify the columns to encode then fit and transform
encoder = ce.backward_difference.BackwardDifferenceEncoder(cols=["engine_type"])
encoder.fit(obj_df, verbose=1)

# Only display the first 8 columns for brevity
encoder.transform(obj_df).iloc[:,8:14].head()

"""
The interesting thing is that you can see that the result are not the 
standard 1’s and 0’s we saw in the earlier encoding examples.
"""

"""
If we try a polynomial encoding, we get a different distribution of values used to encode the columns:
"""
encoder = ce.polynomial.PolynomialEncoder(cols=["engine_type"])
encoder.fit(obj_df, verbose=1)
encoder.transform(obj_df).iloc[:,8:14].head()

"""
There are several different algorithms included in this package and the best way to learn is to try them out 
and see if it helps you with the accuracy of your analysis. 
The code shown above should give you guidance on how to plug in the other approaches 
and see what kind of results you get.
"""

