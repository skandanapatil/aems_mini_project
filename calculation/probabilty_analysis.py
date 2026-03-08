import pandas as pd

def prob(df):
    """
    Performs probability analysis on the Telco Customer Churn dataset:
    - Basic probability
    - Conditional probability
    - Bayes' theorem
    Prints results with interpretations.
    """

    # Ensure TotalCharges is numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    total_customers = len(df)
    print(f"Total customers: {total_customers}\n")

    # -------------------------
    # 1. Basic Probability
    # -------------------------
    p_churn = len(df[df['Churn'] == 'Yes']) / total_customers
    p_monthly = len(df[df['Contract'] == 'Month-to-month']) / total_customers

    print("----- Basic Probability -----")
    print(f"P(Churn) = {p_churn:.2f} → {p_churn*100:.1f}% of all customers churn")
    print(f"P(Month-to-month Contract) = {p_monthly:.2f} → {p_monthly*100:.1f}% of customers on Month-to-month\n")

    # -------------------------
    # 2. Conditional Probability
    # P(Churn | Month-to-month)
    # -------------------------
    monthly_customers = df[df['Contract'] == 'Month-to-month']
    p_churn_given_monthly = len(monthly_customers[monthly_customers['Churn'] == 'Yes']) / len(monthly_customers)

    print("----- Conditional Probability -----")
    print(f"P(Churn | Month-to-month) = {p_churn_given_monthly:.2f} → {p_churn_given_monthly*100:.1f}% of month-to-month customers churn\n")

    # -------------------------
    # 3. Bayes' Theorem
    # P(Month-to-month | Churn)
    # -------------------------
    # Using Bayes theorem: P(A|B) = P(B|A) * P(A) / P(B)
    p_monthly_given_churn = (p_churn_given_monthly * p_monthly) / p_churn

    print("----- Bayes' Theorem -----")
    print(f"P(Month-to-month | Churn) = {p_monthly_given_churn:.2f} → {p_monthly_given_churn*100:.1f}% of churned customers are month-to-month\n")

    # -------------------------
    # Optional: Probability for multiple features
    # -------------------------
    features = ['PaymentMethod', 'SeniorCitizen']
    for feature in features:
        print(f"----- Conditional Probability for {feature} -----")
        for val in df[feature].unique():
            subset = df[df[feature] == val]
            p_val = len(subset) / total_customers
            p_churn_given_val = len(subset[subset['Churn'] == 'Yes']) / len(subset)
            print(f"P(Churn | {feature}={val}) = {p_churn_given_val:.2f} ({p_churn_given_val*100:.1f}%)")
        print()

    print("Probability analysis completed successfully!")