# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('insurance.csv')

# Dimensions of our dataset
dimensions = dataset.shape

# Data type of each atribute
data_types = dataset.dtypes

# Statistical Summary
data_description = dataset.describe()
data_info = dataset.info()

# Categorical data
value_counts_region = dataset['region'].value_counts()

# Class Distribution
data_class_sex = dataset.groupby('sex').size()
data_class_smoker = dataset.groupby('smoker').size()
data_class_region = dataset.groupby('region').size()

# Pairwise Pearson Correlation
correlation = dataset.corr(method='pearson')

# Skew for numeric attribute
data_skew_age = dataset['age'].skew()
data_skew_bmi = dataset['bmi'].skew()
data_skew_children = dataset['children'].skew()
data_skew_charges = dataset['charges'].skew()

# Univariate Histograms
f, axes = plt.subplots(2, 2, figsize=(14,14))
sns.distplot(dataset['age'], ax = axes[0,0])
sns.distplot(dataset['bmi'], ax = axes[0,1])
sns.distplot(dataset['children'], ax = axes[1,0])
sns.distplot(dataset['charges'], ax = axes[1,1])

f, axes = plt.subplots(2, 2, figsize=(14,14))
sns.boxplot(dataset['age'], ax = axes[0,0])
sns.boxplot(dataset['bmi'], ax = axes[0,1])
sns.boxplot(dataset['children'], ax = axes[1,0])
sns.boxplot(dataset['charges'], ax = axes[1,1])

# Boxplot for class
plt.figure(1)
plt.subplots(figsize=(20,20))
plt.subplot(421)
sns.boxplot(x='sex', y='age', data=dataset)
plt.title('Distribución de Edad según Sexo')
plt.grid(True)

plt.subplot(422)
sns.boxplot(x='sex', y='charges', data=dataset)
plt.title('Distribución de Primas según Sexo')
plt.grid(True)

plt.subplot(423)
sns.boxplot(x='region', y='charges', data=dataset)
plt.title('Distribución de Primas según Región')
plt.grid(True)

plt.subplot(424)
sns.boxplot(x='smoker', y='charges', data=dataset)
plt.title('Distribución de Primas según Fumador')
plt.grid(True)

