import pandas as pd
import numpy as np

def preprocess_diabetes_data(df):
    # Make a copy to avoid modifying the original DataFrame in place outside the function scope
    df_processed = df.copy()

    # List of columns where 0s are implausible and represent missing values
    columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Replace 0s with NaN in the specified columns
    df_processed[columns_to_impute] = df_processed[columns_to_impute].replace(0, np.nan)

    # Impute NaN values with the median of each column, avoiding inplace=True warning
    for col in columns_to_impute:
        median_val = df_processed[col].median()
        df_processed[col] = df_processed[col].fillna(median_val)
        print(f"Column '{col}' - Imputed NaN values with median: {median_val}")

    print("Data preprocessing function executed successfully: Implausible 0s handled and NaNs imputed.")
    return df_processed
