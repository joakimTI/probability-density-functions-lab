# The Probability Density Function - Lab

## Introduction
In this lab, we will look at building visualizations known as **density plots** to estimate the probability density for a given set of data. 

## Objectives

You will be able to:

* Plot and interpret density plots and comment on the shape of the plot
* Estimate probabilities for continuous variables by using interpolation 


## Let's get started

Let's import the necessary libraries for this lab.
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import pandas as pd
import seaborn as sns
## Import the data, and calculate the mean and the standard deviation

- Import the dataset 'weight-height.csv' as a pandas dataframe.

- Next, calculate the mean and standard deviation for weights and heights for men and women individually. You can simply use the pandas `.mean()` and `.std()` to do so.

**Hint**: Use your pandas dataframe subsetting skills like `loc()`, `iloc()`, and `groupby()`
data = pd.read_csv('weight-height.csv')
male_df =  data.loc[data['Gender'] == 'Male']
female_df =  data.loc[data['Gender'] == 'Female']

Male_Height_mean = male_df['Height'].mean()
Male_Height_sd = male_df['Height'].std()
Male_Weight_mean = male_df['Weight'].mean()
Male_Weight_sd = male_df['Weight'].std()
Female_Height_mean = female_df['Height'].mean()
Female_Height_sd = female_df['Height'].std()
Female_Weight_mean = female_df['Weight'].mean()
Female_Weight_sd = female_df['Weight'].std()

print('Male Height mean:', Male_Height_mean)
print('Male Height sd:', Male_Height_sd)      

print('Male Weight mean:', Male_Weight_mean)
print('Male Weight sd:' , Male_Weight_sd)   

print('Female Height mean:', Female_Height_mean)
print('Female Height sd:' , Female_Height_sd)      

print('Female Weight mean:', Female_Weight_mean)
print('Female Weight sd:' , Female_Weight_sd)   

# Male Height mean: 69.02634590621737
# Male Height sd: 2.8633622286606517
# Male Weight mean: 187.0206206581929
# Male Weight sd: 19.781154516763813
# Female Height mean: 63.708773603424916
# Female Height sd: 2.696284015765056
# Female Weight mean: 135.8600930074687
# Female Weight sd: 19.022467805319007
## Plot histograms (with densities on the y-axis) for male and female heights 

- Make sure to create overlapping plots
- Use binsize = 10, set alpha level so that overlap can be visualized
fig, ax = plt.subplots(figsize = (8 , 6))
ax.hist(male_df['Height'], bins=10, density = True, alpha = 0.7, label = 'Male Height')
ax.hist(female_df['Height'], bins=10, density = True, alpha = 0.7, label = 'Female Height')
# Increase the number of x-axis tick marks
ticks = np.arange(54, 80 , 2)  # Adjust the range and step size as needed
ax.set_xticks(ticks)
plt.legend()
plt.show()
# Record your observations - are these inline with your personal observations?
Normal distribution with most heights clustered close to the mean. Male are taller that female. There's a close to 13% chance that the height for females is between 62 and 66, for males between 67 and 71.
# Record your observations - are these inline with your personal observations?

# Men tend to have higher values of heights in general than female
# The most common region for male and female heights is between 65 - 67 inches (about 5 and a half feet)
# Male heights have a slightly higher spread than female heights, hence the male height peak is slightly smaller than female height
# Both heights are normally distributed
## Create a density function using interpolation


- Write a density function density() that uses interpolation and takes in a random variable
- Use `np.histogram()`
- The function should return two lists carrying x and y coordinates for plotting the density function
def density(x):
    
    n, bins = np.histogram(x, 20, density=1)
    # Initialize numpy arrays according to number of bins with zeros to store interpolated values
    pdfx = np.zeros(n.size)
    pdfy = np.zeros(n.size)

    # Interpolate through histogram bins 
    # identify middle point between two neighboring bins, in terms of x and y coords
    for k in range(n.size):
        pdfx[k] = 0.5*(bins[k]+bins[k+1])
        pdfy[k] = n[k]
    return pdfx, pdfy
# Generate test data and test the function - uncomment to run the test
np.random.seed(5)
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 100)
x,y = density(s)
plt.plot(x,y, label = 'test')
plt.legend()
plt.show()
## Add overlapping density plots to the histograms plotted earlier
plt.figure(figsize=(7,5))
plt.hist(s, bins = 20, density=True, label = 'Normalized histogram', alpha = 0.7)
# plot the calculated curve

plt.plot(x, y, label = 'Density function')
plt.ylabel ('Probabilities')
plt.legend()
plt.title ('PDF for data')
plt.show()
## Repeat the above exercise for male and female weights
x_m_weight , y_m_weight = density(male_df['Weight'])
x_f_weight , y_f_weight = density(female_df['Weight'])
plt.figure(figsize=(10,8))
plt.hist(male_df['Weight'], bins = 20, density=True, label = 'Male normalized histogram', alpha = 0.7)
plt.hist(female_df['Weight'], bins = 20, density=True, label = 'Female normalized histogram', alpha = 0.7)
# plot the calculated curve

plt.plot(x_m_weight, y_m_weight, label = 'Density function')
plt.plot(x_f_weight, y_f_weight, label = 'Density function')
plt.ylabel ('Probabilities')
plt.legend()
plt.title ('PDF for weight data')
plt.show()
## Write your observations in the cell below
# Record your observations - are these inline with your personal observations?
Normal distribution for both genders. Male are heavier than female. Males are clustered at 185 and females at 140.

# What is the takeaway when comparing male and female heights and weights?
# Record your observations - are these inline with your personal observations?

# The patterns and overlap are highly similar to what we see with height distributions
# Men generally are heavier than women
# The common region for common weights is around 160 lbs. 
# Male weight has slightly higher spread than female weight (i.e. more variation)
# Most females are around 130-140 lbs whereas most men are around 180 pounds.

#Takeaway

# Weight is more suitable to distinguish between males and females than height
## Repeat the above experiments in seaborn and compare with your results
# Code for heights here
sns.displot(male_df.Weight)
sns.displot(female_df.Weight)
plt.title('Comparing Weights')
plt.show()
sns.displot(male_df.Height)
sns.displot(female_df.Height)
plt.title('Comparing Heights')
plt.show()
# Your comments on the two approaches here. 
# are they similar? what makes them different if they are?
# <!-- First approach is better. -->
# Well, what do you think? Overlapping or side to side (or rather top/bottom)
Overlapping is better, it gives a better overview of the two groups with same axes.
## Summary

In this lesson, you learned how to build the probability density curves visually for a given dataset and compare the distributions visually by looking at the spread, center, and overlap. This is a useful EDA technique and can be used to answer some initial questions before embarking on a complex analytics journey.
