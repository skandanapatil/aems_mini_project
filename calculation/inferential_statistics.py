import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def inf_statistics(df):
    """
    Performs inferential statistics on Telco Customer Churn dataset:
    - Confidence intervals for mean and proportion
    - Two-sample t-test for MonthlyCharges (Churn vs No Churn)
    - Chi-square test for Contract vs Churn
    - One-way ANOVA for MonthlyCharges across Contract types
    - Two-way ANOVA for MonthlyCharges by Contract and SeniorCitizen
    """

    # Ensure numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    n = len(df)

    print(f"Total customers: {n}\n")

    # -------------------------
    # 1. Confidence Interval for MonthlyCharges Mean
    # -------------------------
    mean_mc = df['MonthlyCharges'].mean()
    std_mc = df['MonthlyCharges'].std()
    ci_95 = stats.t.ppf(0.975, df=n-1) * (std_mc / np.sqrt(n))
    print(f"95% CI for MonthlyCharges mean: ({mean_mc - ci_95:.2f}, {mean_mc + ci_95:.2f})")

    # -------------------------
    # 2. Confidence Interval for Churn Proportion
    # -------------------------
    p_churn = (df['Churn']=='Yes').mean()
    z = stats.norm.ppf(0.975)
    margin_error = z * np.sqrt(p_churn*(1-p_churn)/n)
    print(f"95% CI for Churn proportion: ({p_churn - margin_error:.2f}, {p_churn + margin_error:.2f})\n")

    # -------------------------
    # 3. Two-sample t-test: MonthlyCharges vs Churn
    # -------------------------
    churned = df[df['Churn']=='Yes']['MonthlyCharges']
    not_churned = df[df['Churn']=='No']['MonthlyCharges']
    t_stat, p_value = stats.ttest_ind(churned, not_churned, equal_var=False)
    print("----- T-Test: MonthlyCharges (Churn vs No Churn) -----")
    print(f"T-statistic = {t_stat:.2f}, P-value = {p_value:.4f}")
    if p_value < 0.05:
        print("MonthlyCharges differ significantly between churned and non-churned customers.\n")
    else:
        print("No significant difference in MonthlyCharges between churned and non-churned customers.\n")

    # -------------------------
    # 4. Chi-square test: Contract vs Churn
    # -------------------------
    contingency = pd.crosstab(df['Contract'], df['Churn'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print("----- Chi-Square Test: Contract vs Churn -----")
    print(f"Chi-square = {chi2:.2f}, P-value = {p:.4f}")
    if p < 0.05:
        print("Contract type is significantly associated with Churn.\n")
    else:
        print("No significant association between Contract type and Churn.\n")

    # -------------------------
    # 5. One-way ANOVA: MonthlyCharges vs Contract
    # -------------------------
    contract_types = df['Contract'].unique()
    groups = [df[df['Contract']==ct]['MonthlyCharges'] for ct in contract_types]
    f_stat, p_value = stats.f_oneway(*groups)
    print("----- One-way ANOVA: MonthlyCharges by Contract Type -----")
    print(f"F-statistic = {f_stat:.2f}, P-value = {p_value:.4f}")
    if p_value < 0.05:
        print("MonthlyCharges differ significantly across contract types.\n")
    else:
        print("No significant difference in MonthlyCharges across contract types.\n")

    # -------------------------
    # 6. Two-way ANOVA: MonthlyCharges vs Contract & SeniorCitizen
    # -------------------------
    model = ols('MonthlyCharges ~ C(Contract) + C(SeniorCitizen) + C(Contract):C(SeniorCitizen)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("----- Two-way ANOVA: MonthlyCharges by Contract & SeniorCitizen -----")
    print(anova_table, "\n")

    print("Inferential statistics analysis completed successfully!")