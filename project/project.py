import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Set plot style
sns.set(style="whitegrid")


# Load the dataset
df = pd.read_csv("C:\\Users\\PRAKASH\\Downloads\\archive (6)\\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Display the first few rows of the dataset
df.head()


# Basic info about the dataset
df.info()

# Summary statistics
df.describe(include='all')


# Plot distribution of numerical variables
df.hist(bins=20, figsize=(20, 15), color='steelblue')
plt.tight_layout()
plt.show()


# Plot distribution of categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=df[col], palette='Set2')
    plt.title(f'Distribution of {col}')
    plt.show()


# Summary statistics for categorical variables
df[categorical_columns].describe()


# Define numerical and categorical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=['object']).columns


# KDE plots for numerical features
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(df[col], shade=True, color='green')
    plt.title(f'Density Plot of {col}')
    plt.show()


# Boxplot for each numerical feature
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(df[col], color='orange')
    plt.title(f'Boxplot of {col}')
    plt.show()


# Correlation heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


# Countplots of categorical variables vs Attrition
for col in categorical_columns:
    if col != 'Attrition':
        plt.figure(figsize=(8, 4))
        sns.countplot(y=df[col], hue=df['Attrition'], palette='Set1')
        plt.title(f'{col} vs Attrition')
        plt.show()


# Boxplots of numerical variables vs Attrition
numerical_columns = df.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Attrition', y=col, data=df, palette='Set3')
    plt.title(f'{col} vs Attrition')
    plt.show()


# Scatterplots of numerical features
for i in range(len(numerical_columns)):
    for j in range(i+1, len(numerical_columns)):
        plt.figure(figsize=(8, 4))
        sns.scatterplot(x=df[numerical_columns[i]], y=df[numerical_columns[j]], hue=df['Attrition'], palette='Set1')
        plt.title(f'{numerical_columns[i]} vs {numerical_columns[j]}')
        plt.show()


# Display the column names to confirm the exact name of the target variable
print(df.columns)


# Violin plots for numerical variables by Attrition
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.violinplot(x='Attrition', y=col, data=df, palette='Set2')
    plt.title(f'Violin Plot of {col} by Attrition')
    plt.show()


# Pairplot of selected numerical features colored by Attrition
selected_columns = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany', 'Attrition']
sns.pairplot(df[selected_columns], hue='Attrition', palette='Set1')
plt.show()


# Cross-tabulation and Chi-Square test
for col in categorical_columns:
    if col != 'Attrition':
        contingency_table = pd.crosstab(df['Attrition'], df[col])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f'Chi-Square Test for {col}')
        print(f'Chi2: {chi2}, p-value: {p}\n')


# Pairplot of selected features with hue as Attrition
selected_columns = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany', 'Attrition']
sns.pairplot(df[selected_columns], hue='Attrition', palette='husl')
plt.show()


# Grouped bar charts for categorical features
for col in categorical_columns:
    if col != 'Attrition':
        plt.figure(figsize=(12, 6))
        sns.countplot(x=df[col], hue=df['Attrition'], palette='coolwarm')
        plt.title(f'{col} vs Attrition')
        plt.xticks(rotation=45)
        plt.show()


# Heatmap of categorical variables using crosstab
for col in categorical_columns:
    if col != 'Attrition':
        ct = pd.crosstab(df[col], df['Attrition'])
        sns.heatmap(ct, annot=True, cmap='YlGnBu', linewidths=0.5)
        plt.title(f'Heatmap of {col} vs Attrition')
        plt.show()


from mpl_toolkits.mplot3d import Axes3D

# 3D Scatter plot for three numerical features
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Age'], df['MonthlyIncome'], df['YearsAtCompany'], c=df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0), cmap='coolwarm')
ax.set_xlabel('Age')
ax.set_ylabel('Monthly Income')
ax.set_zlabel('Years at Company')
plt.title('3D Scatter Plot of Age, Monthly Income, and Years at Company')
plt.show()


# Checking for missing values
df.isnull().sum()

# If there are missing values, you can handle them as follows:
# df.fillna(method='ffill', inplace=True)  # Forward fill for missing values


# Checking high cardinality in categorical features
for col in categorical_columns:
    print(f"{col} has {df[col].nunique()} unique values.")


# Example of creating new features
df['TotalWorkingYearsRatio'] = df['YearsAtCompany'] / df['TotalWorkingYears']
df['IncomePerYear'] = df['MonthlyIncome'] * 12


# Analyzing new features
plt.figure(figsize=(8, 4))
sns.boxplot(x='Attrition', y='IncomePerYear', data=df, palette='Set2')
plt.title('Income Per Year vs Attrition')
plt.show()

plt.figure(figsize=(8, 4))
sns.violinplot(x='Attrition', y='TotalWorkingYearsRatio', data=df, palette='Set1')
plt.title('Total Working Years Ratio vs Attrition')
plt.show()
