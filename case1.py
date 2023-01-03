#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly
import plotly.io as pio

pio.renderers.default = "notebook"


# Enable Plotly in Jupyter notebooks
plotly.offline.init_notebook_mode()

# Read the CSV file
df = pd.read_csv("loans_full_schema.csv")

pd.set_option("display.max_columns", None)


# In[2]:


pd.set_option('display.max_rows',  df.shape[0]+1)
# missing = df.isnull().sum()


# # Dataset Describtion & Issues
# The Lending Club data set contains information on thousands of loans made through the Lending Club platform. It includes a variety of features related to the borrower, such as their employment status, credit score, and income, as well as features related to the loan, such as the loan amount, interest rate, and term. There are 55 variables in total, including both numerical and categorical variables.
# 
# There are a few potential issues with this data set.
# First off al there are missing values some variables, however they are not going to be used for the data exploration and visualization.
# Another issue is that it only represents loans that were actually made, not all loan applications.
# This means that it is possible that some of the riskier borrowers were not included in the data set
# because they were not offered a loan or did not accept a loan offer due to a high interest rate.
# Another issue is that the data may be biased towards borrowers with higher
# credit scores, since these borrowers may be more likely to be approved for a loan and to accept a loan offer.
# These issues should be kept in mind when interpreting the results of any analysis or visualizations using this data.

# In[3]:


import plotly_express as px
import plotly.graph_objects as go

# Count the number of loans for each purpose
purposes = df['loan_purpose'].value_counts()

# Calculate the percentages
percentages = purposes / purposes.sum()

# Sort the values in descending order
percentages = percentages.sort_values(ascending=False)

# Create the pie chart
fig = px.pie(values=percentages, names=percentages.index, labels={'value': 'Percentage', 'name': 'Loan Purpose'})

# Update the title and font
fig.update_layout(title_text='Loan Purposes', titlefont={'size': 20})

# Add a legend
fig.update_layout(legend=dict(title='Loan Purpose', yanchor='top', y=1.02, xanchor='left', x=1))

fig.show()


# # Observations
# According to the percentages, it appears that the most common purpose for loans in this dataset is debt consolidation, followed by credit card refinancing, other purposes, and home improvement. It may be advisable for the lending platform to focus its advertising efforts on these commonly requested loan purposes, as they likely represent a significant portion of the market.

# In[4]:


import plotly.express as px

# Group the data by loan purpose and calculate the mean interest rate for each group
grouped = df.groupby('loan_purpose')['interest_rate'].mean()

# Sort the values in ascending order
grouped = grouped.sort_values()

# Get the names of the loan purposes and the mean interest rates
purposes = grouped.index
rates = grouped.values

# Create the histogram
fig = px.bar(x=purposes, y=rates) #, color=rates, color_continuous_scale='Viridis_R')
fig.update_layout(title_text='Average Interest Rate by Loan Purpose', yaxis_title="Avg Interest Rate", xaxis_title = "Loan Purpose")

fig.show()


# # Observations
# It appears that loan borrowers are willing to accept higher interest rates for loans with the purpose of financing renewable energy projects. This may be because the long-term cost savings associated with renewable energy outweigh the short-term burden of a higher interest rate. In other words, the higher interest rate may not have a negative impact on the overall financial situation of the borrower.
# 
# Additionally, it is notable that debt consolidation loans tend to have relatively high interest rates. This is likely due to the fact that a large proportion of loan borrowers use their loans for the purpose of debt consolidation. As such, lenders may be able to charge higher interest rates and still attract a sufficient number of borrowers. Furthermore, it is possible that borrowers seeking debt consolidation loans may be perceived as higher risk by lenders, leading to higher interest rates as a means of compensation for this risk.

# In[5]:


# average interest_rate per home_ownership 
grouped = df.groupby('homeownership')['interest_rate'].mean()

# Sort the values in ascending order
grouped = grouped.sort_values()

# Get the names of the home ownership categories and the mean interest rates
ownerships = grouped.index
rates = grouped.values

# Create the histogram
fig = px.bar(x=ownerships, y=rates) #, color=rates, color_continuous_scale='Viridis_R')
fig.update_layout(title_text='Average Interest Rate by Home Ownership', yaxis_title="Avg Interest Rate", xaxis_title = "Home Ownership")

# Show the plot
fig.show()


# # Observations
# It appears that renters may have a higher average interest rate compared to homeowners, which could be due to a variety of factors such as creditworthiness, income, or the lender's risk assessment. It is important to note that the data for homeowners who own their homes outright is relatively scarce in this dataset. Specifically, only about 1.3k observations have the value "OWN" for home ownership, while the majority (approximatey half) of the data have "MORTGAGE" as the value. Given this imbalance, it may not be reliable to draw conclusions about the relationship between home ownership and interest rates based solely on this dataset..

# In[6]:


state_rates = df.groupby('state')['interest_rate'].mean()
state_rates_df = pd.DataFrame({"state":state_rates.index, "interest_rate":state_rates.values})
fig = px.choropleth(state_rates_df,
                    locations='state', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='interest_rate',
                    color_continuous_scale="YlOrRd",
                    title='Average Interest Rate by State',
                    )

fig.show()


# # Observations
# It is noteworthy that the interest rates in North Dakota and Huawei are among the highest in the market. Lender may be able to capitalize on this information by offering competitive rates and terms in these regions.

# In[7]:


loans_per_state = df.groupby('state').size()

# Convert the Series to a DataFrame
loans_per_state_df = loans_per_state.to_frame(name='number_of_loans')

# Reset the index of the DataFrame to move the state column to the data
loans_per_state_df = loans_per_state_df.reset_index()

fig = px.choropleth(loans_per_state_df,
                    locations='state', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='number_of_loans',
                    color_continuous_scale="YlGnBu",
                    title='Number of loans per State',
                    hover_name='state')
fig.show()


# # Observations 
# California, Texas, New York, and Florida have a significantly higher volume of loan requests compared to other states. Lenders utilizing this information to their advantage can be beneficial. One way to do this is to focus marketing efforts in these states, as they are likely to have a larger pool of potential borrowers. Additionally, by understanding the local lending market in these states, lenders can offer competitive rates and terms to attract borrowers and increase the likelihood of loan approval.

# In[8]:


# Set up the data for the chart
loan_amount = df['emp_length']
interest_rate =  df['interest_rate']

# Create a 2D histogram with the loan amount on the X-axis and the interest rate on the Y-axis
fig = go.Figure(data=[go.Histogram2d(x=loan_amount, y=interest_rate, colorscale='Viridis')])

# Add axis labels and a title
fig.update_layout(xaxis_title='Years in job', yaxis_title='Interest Rate', title='Interest Rates by Years in job')

fig.show()


# # Observations
# The heat map above indicates that the majority of loans are extended to individuals who have been employed for at least 10 years and are offered at competitively low interest rates. As such, it may be beneficial for the lending platform to focus its advertising efforts on this demographic and emphasize the favorable interest rates available on the platform.

# In[9]:


"""
    sum of missing values per column
debt_to_income                        24
num_accounts_120d_past_due           318
emp_title                            833 should be dropped
emp_length                           817 should be dropped   
annual_income_joint                 8505 should be dropped
verification_income_joint           8545 should be dropped
debt_to_income_joint                8505 should be dropped
months_since_last_delinq            5658 should be dropped
months_since_90d_late               7715 should be dropped
months_since_last_credit_inquiry    1271 should be dropped

since we only have 10000 examples we cannot afford to remove much data with missing values,
thus we should drop columns where a lot of data is missing
"""

# drop columns with a lot of missing values
# removes rows with missing values on debt_to_income and num_accounts_120d_past_due
# maps categorical features to float values
def get_clean_set(df):
    # create a copy of the initial dataset
    new_df = df.copy()
    # columns where there are a lot of missing values, thus should be dropped
    columns_to_drop = ['annual_income_joint', 'verification_income_joint', 'debt_to_income_joint', 
                   'months_since_last_delinq', 'months_since_90d_late', 'months_since_last_credit_inquiry', 'emp_title', 'emp_length']
    #drop columns
    new_df = new_df.drop(columns_to_drop, axis=1)
    
    #drop rows with missing values on debt_to_income and num_accounts_120d_past_due 
    new_df = new_df.dropna(subset=['debt_to_income', 'num_accounts_120d_past_due'])
    
    #  maps categorical features to float values
    state_mapping = {'NJ': 1.0,  'HI':  2.0, 'WI':  3.0, 'PA':  4.0, 'CA':  5.0, 'KY':  6.0, 'MI':  7.0, 'AZ':  8.0, 'NV': 9.0,  'IL': 10.0, 'FL': 11.0,
                     'SC': 12.0, 'CO': 13.0, 'TN': 14.0, 'TX': 15.0, 'VA': 16.0, 'NY': 17.0, 'GA': 18.0, 'MO': 19.0, 'AR': 20.0, 'MD': 21.0, 'NC': 22.0,
                     'NE': 23.0, 'WV': 24.0, 'NH': 25.0, 'UT': 26.0, 'DE': 27.0, 'MA': 28.0, 'OR': 29.0, 'OH': 30.0, 'OK': 31.0, 'SD': 32.0, 'MN': 33.0,
                     'AL': 34.0, 'WY': 35.0, 'LA': 36.0, 'IN': 37.0, 'KS': 38.0, 'MS': 39.0, 'WA': 40.0, 'ME': 41.0, 'VT': 42.0, 'CT': 43.0, 'NM': 44.0,
                     'AK': 45.0, 'MT': 46.0, 'RI': 47.0, 'ND': 48.0, 'DC': 49.0, 'ID': 50.0}
    
    home_ownership_mapping = {'MORTGAGE':1.0, 'RENT':2.0, 'OWN':3.0}
    
    verified_income_mapping = {'Verified':1.0, 'Not Verified':2.0, 'Source Verified':3.0}
    
    loan_purpose_mapping = {'moving':0, 'debt_consolidation':1, 'other':2, 'credit_card':3.0,
       'home_improvement':4.0, 'medical':5.0, 'house':6.0, 'small_business':7.0, 'car':8.0,
       'major_purchase':9.0, 'vacation':10.0, 'renewable_energy':11.0}
    
    app_type_mapping = {'individual':0, 'joint':1}
    
    grade_mapping = {'C':3.0, 'D':4.0, 'A':1.0, 'B':2.0, 'F':6.0, 'E':5.0, 'G':7.0}
    
    subgrade_mapping = {'C3':13.0, 'C1':11.0, 'D1':16.0, 'A3':3.0, 'C2':12.0, 'B5':10.0,
                        'C4':14.0, 'B2':7.0, 'B1':6.0, 'D3':18.0, 'F1':21.0,  'E5':20.0,
                        'A2':2.0, 'A5':5.0, 'A4':4.0, 'A1':1.0, 'D4':19.0, 'D5':20.0,
                        'B3':8.0, 'D2':17.0, 'E1':16.0, 'G1':26.0, 'B4':9.0, 'C5':15.0,
                        'E2':17.0, 'E4':19.0, 'F3':23.0, 'E3':18.0, 'F5':25.0, 'F2':22.0,
                        'F4':24.0, 'G4':27.0}
    
    init_list_status_mapping = {'whole':1.0, 'fractional':2.0}
    
    disbursement_method_mapping = {'Cash':0.0, 'DirectPay':2.0}
    
    loan_status_mapping = {'Current':0.0, 'Fully Paid':1.0, 'In Grace Period':2.0, 'Late (31-120 days)':3.0,
       'Charged Off':4.0, 'Late (16-30 days)':5.0}
    
    # replace categorical values with float values
    new_df['state'] = df['state'].map(state_mapping)
    new_df['homeownership'] = df['homeownership'].map(home_ownership_mapping)
    new_df['verified_income'] = df['verified_income'].map(verified_income_mapping)
    new_df['loan_purpose'] = df['loan_purpose'].map(loan_purpose_mapping)
    new_df['application_type'] = df['application_type'].map(app_type_mapping)
    new_df['grade'] = df['grade'].map(grade_mapping)
    new_df['sub_grade'] = df['sub_grade'].map(subgrade_mapping)
    new_df['initial_listing_status'] = new_df['initial_listing_status'].map(init_list_status_mapping)
    new_df['disbursement_method'] = new_df['disbursement_method'].map(disbursement_method_mapping)
    new_df['loan_status'] = new_df['loan_status'].map(loan_status_mapping)
    
    return new_df


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error



model_df = get_clean_set(df)

features = [
    'state', 'homeownership', 'annual_income', 'verified_income',
    'debt_to_income', 'delinq_2y', 'earliest_credit_line',
    'inquiries_last_12m', 'total_credit_lines', 'open_credit_lines',
    'total_credit_limit', 'total_credit_utilized',
    'num_collections_last_12m', 'num_historical_failed_to_pay',
    'current_accounts_delinq', 'total_collection_amount_ever',
    'current_installment_accounts', 'accounts_opened_24m',
    'num_satisfactory_accounts', 'num_accounts_120d_past_due',
    'num_accounts_30d_past_due', 'num_active_debit_accounts',
    'total_debit_limit', 'num_total_cc_accounts', 'num_open_cc_accounts',
    'num_cc_carrying_balance', 'num_mort_accounts',
    'account_never_delinq_percent', 'tax_liens', 'public_record_bankrupt',
    'loan_purpose', 'application_type', 'loan_amount', 'term']

X_train, X_test, y_train, y_test = train_test_split(model_df[features], model_df['interest_rate'], test_size=0.1)

# build the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# build the random forest model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# build the elastic net model
en = ElasticNet(l1_ratio=0.5, alpha=0.1)
en.fit(X_train, y_train)

# test the elastic net model
en_pred = en.predict(X_test)
en_mse = mean_squared_error(y_test, en_pred)
en_score = en.score(X_test, y_test)

# test the linear regression model
lr_pred = lr.predict(X_test)
lr_score = lr.score(X_test, y_test)
lr_mse = mean_squared_error(y_test, lr_pred)

# build the random forest model
rf_pred = rf.predict(X_test)
rf_score = rf.score(X_test, y_test)
rf_mse = mean_squared_error(y_test, rf_pred)


# # Data cleaning
# The columns 'emp_title', 'emp_length', 'annual_income_joint', 'verification_income_joint', 'debt_to_income_joint', 'months_since_last_delinq', 'months_since_90d_late', and 'months_since_last_credit_inquiry' were missing from 800 to 8000 values and the dataset has only 10k examples. Therefore, these columns were removed. In addition, data rows with missing values on 'debt_to_income' and 'num_accounts_120d_past_due' were also removed. Categorical values were transformed into numerical ones.

# # Assumptions
# 
# In order to predict the interest rate of a loan prior to its issuance, certain features such as 'grade' and 'paid_total', which are indicators of the interest rate, were removed. Regression models from the scikit-learn library were employed as the problem at hand is a regression problem. The performance of the models was evaluated using mean squared error and the score provided by scikit-learn.

# In[11]:


model_names = ['Linear Regression', 'Elastic Net', 'Random Forests']
scores = [lr_score*100, en_score*100,  rf_score*100]
mse = [lr_mse, en_mse,  rf_mse]

# Create a bar chart with the scores as the Y-values and the model names as the X-values
fig = go.Figure(data=[go.Bar(x=model_names, y=scores, name='Scores'),
                      go.Bar(x=model_names, y=mse, name='MSE')])

# Add axis labels and a title
fig.update_layout(xaxis_title='Model', yaxis_title='ScoreMSE', title='Model Scores and MSEs')

# Show the chart
fig.show()


# # Conclusion & Future Model Enhancements
# In the above analysis, we compare the mean squared error and score of each model. The Random Forests model performs better than the other two models due to the complexity of the dataset, which contains both categorical and continuous features. Additionally, there may not be sufficient data to accurately train the models and there are a large number of features, making it difficult to determine which have the greatest impact on the interest rate. If more time were available, further visualization of the data could be conducted to identify the most suitable model for this problem and experimentation with the hyperparameters could be performed to improve model performance.
# 
