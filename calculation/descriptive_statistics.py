import pandas as pd

def descriptive_analysis(df):
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    print("----- CENTRAL TENDENCY -----")
    print("\nMean:")
    print(df[numerical_cols].mean())

    print("\nMedian:")
    print(df[numerical_cols].median())


    print("\n----- DISPERSION -----")

    print("\nVariance:")
    print(df[numerical_cols].var())

    print("\nStandard Deviation:")
    print(df[numerical_cols].std())


    print("\n----- FIVE NUMBER SUMMARY -----")
    print(df[numerical_cols].describe().loc[['min','25%','50%','75%','max']])

    print("\n----- IQR AND OUTLIERS -----")

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        print(f"\nColumn: {col}")
        print("IQR:", IQR)
        print("Lower Bound:", lower)
        print("Upper Bound:", upper)
        print("Number of Outliers:", outliers.shape[0])