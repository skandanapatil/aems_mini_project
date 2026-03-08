# Identify data type (numerical: continuous/discrete, categorical: nominal/ordinal, 
# time-series, multivariate), specify number of observations and variables, identify 
# target variable(s), display tabular preview of dataset.

# Dataset Description

# The Telco Customer Churn dataset contains information about telecom customers and whether they discontinued the service.

# Observations (rows): 7043

# Variables (columns): 21

# Dataset type: Multivariate dataset with numerical and categorical variables.

# Numerical Variables

# Continuous

# MonthlyCharges

# TotalCharges

# Discrete

# tenure

# Categorical Variables

# Nominal

# gender

# Partner

# Dependents

# InternetService

# PaymentMethod

# Ordinal

# Contract (Month-to-month → One year → Two year)

# Target Variable
# Churn

# Values:

# Yes → customer left

# No → customer stayed

def identify_data_type(df):

    print("Number of observations:", df.shape[0])
    print("Number of variables:", df.shape[1])

    print("\nData types")
    print(df.dtypes)

    print("\nDataset preview")
    print(df.head())