import pandas as pd
from  calculation.data_understanding import identify_data_type
from calculation.descriptive_statistics import descriptive_analysis
from calculation.visualization import generate_all_plots
from calculation.probabilty_analysis import prob
from calculation.inferential_statistics import inf_statistics
df = pd.read_csv("Telco_Customer_Churn.csv")
identify_data_type(df)
descriptive_analysis(df)
generate_all_plots(df)
prob(df)
inf_statistics(df)