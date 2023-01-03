#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import plotly
import plotly_express as px

plotly.offline.init_notebook_mode()


# In[18]:


df = pd.read_csv("casestudy.csv")


# In[19]:


revenue_by_year_df = df.groupby("year")["net_revenue"].sum()

years = revenue_by_year_df.index
revenues = revenue_by_year_df.values

fig = px.bar(x=years, y=revenues)
fig.update_layout(title_text="Revenue by year", yaxis_title="Revenue", xaxis_title = "Year", xaxis=dict(tickmode='linear'))
fig.show()


# In[20]:


"""
    returns a dataframe of  year:net_revenue
    where net_revenue is the total revenue of the new custoemrs for each eyar
"""
def new_customer_revenue_per_year():
    new_customer_revenue = []
    emails_by_year = df.groupby('year')['customer_email'].unique()

    for year, emails in emails_by_year.iteritems():
        new_emails = set(emails) - set(emails_by_year.get(year-1, []))
        new_customers_df = df[df['customer_email'].isin(new_emails) & (df['year'] == year)]
        new_customer_revenue.append((year, new_customers_df['net_revenue'].sum()))    
    return pd.DataFrame(new_customer_revenue, columns=['year', 'net_revenue'])


# In[21]:


new_customer_revenue_df = new_customer_revenue_per_year()

years = new_customer_revenue_df['year']
revenues = new_customer_revenue_df['net_revenue']
fig = px.bar(x=years[1:], y=revenues[1:])
#fig = px.bar(x=years, y=revenues)
fig.update_layout(title_text="Revenue of new customers by year", yaxis_title="Revenue", xaxis_title = "Year", xaxis=dict(tickmode='linear'))
fig.show()


# # Note
# The visualizations below depict data for all years except the first one, as the case study requested information that required data from previous years.

# In[22]:


revenue_temp = 0
growth=[]
for revenue in revenue_by_year_df:
    growth.append(revenue-revenue_temp)
    revenue_temp = revenue
    
fig = px.line(x=years[1:], y=growth[1:])
fig.update_layout(title_text="Growth", yaxis_title="Current-Previous Year revenue", xaxis_title = "Year",xaxis=dict(tickmode='linear')) 
fig.show()


# In[23]:


by_year_df = df.groupby('year')
lost_revenue_years = years[:-1]
lost_revenue = []

# Iterate through each year
for year, year_df in by_year_df:
    if year == 2017: break
    # Identify the customers who made a purchase in the current year but not in the following year
    current_year_customers = set(year_df['customer_email'])
    next_year_customers = set(by_year_df.get_group(year + 1)['customer_email'])
    lost_customers = current_year_customers - next_year_customers
    lost_revenue.append(year_df[year_df['customer_email'].isin(lost_customers)]['net_revenue'].sum())

fig = px.bar(x=lost_revenue_years, y=lost_revenue)
fig.update_layout(title_text="Lost Revenue by Attrition", yaxis_title="Lost revenue", xaxis_title = "Year", xaxis=dict(tickmode='linear'))
fig.show()


# In[24]:


# these are what i am looking for
added_revenue_per_year = []   # list of tuples containing the current and prior year revenue for existing customers
total_customers_per_year = [] # list of tuples containing the number of current and prior year customers
new_customers_per_year = []   # list of integers representing the number of new customers for each year
lost_customers_per_year = []  # list of integers representing the number of lost customers for each year


# In[25]:


#Iterate through each year
by_year_df = df.groupby('year')

for year, year_df in by_year_df:
    
    # skip first year since there are not existing customers
    if year == 2015: continue
        
    # Identify the customers who made a purchase in the current year and in the previous year
    current_year_customers = set(year_df['customer_email'])
    previous_year_customers = set(by_year_df.get_group(year - 1)['customer_email'])
    existing_customers = current_year_customers & previous_year_customers # customer that exist both the prior & current year
    
    prior_df = by_year_df.get_group(year-1)
    # compute current and prior year revenue
    current_year_revenue = year_df[year_df['customer_email'].isin(existing_customers)]['net_revenue'].sum()
    prior_year_revenue = prior_df[prior_df['customer_email'].isin(existing_customers)]['net_revenue'].sum()
    
    # append results to the lists
    added_revenue_per_year.append((current_year_revenue, prior_year_revenue))
    
    # compute customers per year
    total_customers_per_year.append((len(current_year_customers),len(previous_year_customers)))
    
    # compute numner of new and lost customer per year
    new_customers_per_year.append(len(current_year_customers - existing_customers)) 
    lost_customers_per_year.append(len(previous_year_customers - existing_customers))    


# In[26]:


import plotly.express as px

df = pd.DataFrame({'year': years[1:], 
                   'Current Year Revenue': [x[0] for x in added_revenue_per_year], 
                   'Prior Year Revenue': [x[1] for x in added_revenue_per_year]})

title = "Revenue by year for customers present in that year and prior years"
fig = px.bar(df, x='year', y=['Current Year Revenue', 'Prior Year Revenue'], title=title, barmode='group', labels={'y': 'Existing Customer Revenue (USD)', 'x': 'Year'})
fig.update_layout(yaxis_title='Revenue', xaxis=dict(tickmode='linear'))
fig.show()


# In[27]:


import plotly.express as px

df = pd.DataFrame({'year': years[1:], 
                   'Current Year Customers': [x[0] for x in total_customers_per_year], 
                   'Prior Year Customers': [x[1] for x in total_customers_per_year]})

fig = px.bar(df, x='year', y=['Current Year Customers', 'Prior Year Customers'], title='Number of Customers by Year', barmode='group')
fig.update_layout(yaxis_title='Number of Customers', xaxis=dict(tickmode='linear'))
fig.show()


# In[28]:


import plotly.express as px

df = pd.DataFrame({'year': years[1:], 
                   'New Customers': new_customers_per_year, 
                   'Lost Customers': lost_customers_per_year})

fig = px.bar(df, x='year', y=['New Customers', 'Lost Customers'], title='Number of New and Lost Customers by Year', barmode='group')
fig.update_layout(yaxis_title='Number of Customers', xaxis=dict(tickmode='linear'))
fig.show()


# # Observations
# Annually, there is a significant influx and outflux of customers. In particular, the period from 2015 to 2016 saw a decrease in total revenue by 3.3 million, attributed to a decrease in customer numbers. Customers who made purchases in a given year and the preceding year contribute the same amount of revenue in each year. The loss of customers, or customer attrition, also significantly impacts revenue loss on a yearly basis.
