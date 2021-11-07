#!/usr/bin/env python
# coding: utf-8

# # Investment Analysis Assignment
# 
# ### 1.Importing Liberaries required

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt , seaborn as sns


# ### 2.  Reading Data

# In[2]:


companies = pd.read_csv("./data/companies.csv",encoding = "palmos") #Using Palmos as adviced by Upgrad TA in discussion forum
rounds2 = pd.read_csv("./data/rounds2.csv",encoding = "palmos") # Used multiple encoding and then finallised with using palmos
mapping = pd.read_csv("./data/mapping.csv", encoding = "palmos")


# In[3]:


companies.head(3)


# In[4]:


companies.info() # to take the overview of the 'companies' dataframe structure


# In[5]:


rounds2.head(3)


# In[6]:


rounds2.info() # to take the overview of the 'rounds2' dataframe structure


# In[7]:


mapping.head(3)


# In[8]:


mapping.info() # to take the overview of the 'mapping' dataframe structure


# #### *How many Unique values present in round2.csv

# In[9]:


rounds2['company_permalink'].describe()


# #### ** Converting rounds2['company_permalink'] column to lower case because of inproper formating

# In[10]:


rounds2["company_permalink"] = rounds2["company_permalink"].str.lower() #Converting to lower case to handle duplicate values
                                                                        #because of lower and upper case issues


# #### Using describe function to find unique values. len(rounds2['company_permalink'].unique())  can also be used 
#     Using permalink column as a unique name identifier

# In[11]:


print(len(rounds2['company_permalink'].unique())) ## Finding unique company name in rounds2 dataframe 
rounds2['company_permalink'].describe()           ## len(rounds2['company_permalink'].unique())  can also be used
                                                  ## Unique Companies = 66368


# ### Question : How many unique companies are present in rounds2?
#     Answer : 66368
# 

# #### *How many unique companies are present in companies?

# In[12]:


companies["permalink"] = companies["permalink"].str.lower() ## Lower casing the permalink column 
print(companies['permalink'].describe())                     ## Using describe() to find unique values
print(len(companies['permalink'].unique()))                 ## Unique Company names in companies dataframe = 66368 
print(len(companies['name'].unique()))                                                              


# In[13]:


pd.set_option("display.max_rows", None, "display.max_columns", None) # to display whole dataframe
companies[companies.name.duplicated()].sort_values(by=['name'])     # Analysing the difference in unique values of permalink and
                                                                    # name column of companies


# ##### * After analysing the dataframe , the companies with same name are actually different as they are operating in different location and different category so using permalink as the correct column to find correct numbers of unique companies , so we will ignore this

# #### Question: How many unique companies are present in the companies file?
#     Answer: 66368

# In[14]:


print('Total rows ' , companies.shape[0])                    # Find total number of rows
print(len(companies['permalink'].unique()))                  # Looking at unique value count of permalink


# #### Question: In the companies data frame, which column can be used as the  unique key for each company? Write the name of the column.
#     Answer: Permalink

# In[15]:


rounds2.loc[~rounds2['company_permalink'].isin(companies['permalink']), :] # Checking that if there is any value in rounds2 that
                                                                           # is not present in companies


# #### Question: Are there any companies in the rounds2 file which are not  present in companies ? 
#     Answer : No

# ### Merging companies and rounds2 dataframe to master_frame

# In[16]:


master_frame = pd.merge(rounds2,companies, left_on="company_permalink", right_on="permalink", how='left')


# In[17]:


master_frame.shape


# #### Question: Merge the two data frames so that all  variables (columns)  in the companies frame are added to the rounds2 data frame. Name the merged frame master_frame. How many observations are present in master_frame ?
# Answer : 114949
# 

# # Cleaning Master_frame as per requirement of Analysis

# In[18]:


#checking null values
round(100*master_frame.isnull().sum()/len(master_frame.index),2)


# In[19]:


#Since our analysis requires Raised_amount_usd , country_code and category_list , so
#we will remove the data where these columns are null
master_frame = master_frame[~(master_frame['raised_amount_usd'].isnull())]
master_frame = master_frame[~(master_frame['country_code'].isnull())]
master_frame = master_frame[~(master_frame['category_list'].isnull())]


# In[20]:


round(100*master_frame.isnull().sum()/len(master_frame.index),2)


# #### **Funding_round_code , state_code ,founded_at, city , region, homepage_url are not needed so I will ignore the missing values of these columns          

# In[21]:


# checking the percentage of retained rows after cleaning
print(round(100*(len(master_frame.index)/114949),2))#around 22% of the data is removed  (26420 rows)
print(master_frame.shape)


# ### Spark Funds wants to choose among 'seed', 'venture', 'angel' and 'private_equity' investment types for each potential investment they will make.
# ##### So I will remove the rows other than these 'funding_round_type' values

# In[22]:


# Keeping only the rows with funding_round_type as 'venture' or 'seed' or 'angel' or 'private_equity'
master_frame = master_frame[(master_frame['funding_round_type'] == 'venture') 
                            | (master_frame['funding_round_type'] == 'seed')
                            | (master_frame['funding_round_type'] == 'angel')
                            | (master_frame['funding_round_type'] == 'private_equity')]


# In[23]:


print(round(100*(len(master_frame.index)/88529),2)) # another 15% data is removed


# In[24]:


master_frame.groupby(master_frame['funding_round_type'])['raised_amount_usd'].mean()


# In[25]:


# adding new column by converting the values of dollar to million dollar for better understanding
master_frame['raised_amount_usd_mil'] = round(master_frame['raised_amount_usd']/1000000,4)


# In[26]:


master_frame.head()


# In[27]:


master_frame.groupby(master_frame['funding_round_type'])['raised_amount_usd_mil'].mean().sort_values(ascending = False)


# In[28]:


master_frame.groupby(master_frame['funding_round_type'])['raised_amount_usd_mil'].median().plot.bar()
master_frame['funding_round_type'].value_counts() # To find the maximum funding type people invest


# In[29]:


# increase figure size 
plt.figure(figsize=(18, 6))

# subplot 1: statistic=mean
plt.subplot(1, 3, 1)
sns.barplot(x='funding_round_type', y='raised_amount_usd_mil', data=master_frame)
plt.title("Average")
plt.axhline(y=5, linewidth=2)
plt.axhline(y=15, linewidth=2)

# subplot 2: statistic=median
plt.subplot(1, 3, 2)
sns.barplot(x='funding_round_type', y='raised_amount_usd_mil', data=master_frame, estimator=np.median)
plt.title("Median")

#subplot 3: Count of funding in each type
plt.subplot(1, 3, 3)
sns.countplot(x='funding_round_type', data=master_frame)
plt.title("Count")


# In[30]:


master_frame.groupby(master_frame['funding_round_type'])['raised_amount_usd_mil'].describe()


# #### Clearly people are investing most in the venture funding type and it lies between 5  mil and 15 mil

# In[31]:


# boxplot of a variable across various funding categories
plt.figure(figsize=(15, 5))
sns.boxplot(x='funding_round_type', y='raised_amount_usd', data=master_frame)
plt.yscale('log')
plt.show()


# #### Average funding amount of venture type
# 

# In[32]:


round(master_frame[master_frame['funding_round_type']=='venture'].raised_amount_usd.mean(),2)


# #### Question : Average funding amount of venture type 
# Answer : 11724222.69 or 11.72 million

# In[33]:


# Average funding amount of angel type
round(master_frame[master_frame['funding_round_type']=='angel'].raised_amount_usd.mean(),2)


# #### Question : Average funding amount of angel type
# Answer : 971573.89 or 0.97 mil

# In[34]:


#Average funding amount of seed type
round(master_frame[master_frame['funding_round_type']=='seed'].raised_amount_usd.mean(),2)


# #### Question : Average funding amount of seed type
# Answer : 747793.68 or 0.75 mil

# In[35]:


#Average funding amount of private equity type
round(master_frame[master_frame['funding_round_type']=='private_equity'].raised_amount_usd.mean(),2)


# #### Question : Average funding amount of private equity type
# Answer : 73938486.28 or 73.94

# In[36]:


# Considering that Spark Funds wants to invest between 5 to 15 million USD per  investment round,
# which investment type is the most suitable for them?
# Using the investment count per round type and mean and median of each round type , Venture is the appropriate choice for
# investment.


# #### Question : Considering that Spark Funds wants to invest between 5 to 15 million USD per  investment round, which investment type is the most suitable for them?
# Answer : Venture

# In[37]:


# Removing rows other than Venture funding type
master_frame = master_frame[master_frame['funding_round_type'].isin(['venture'])]


# In[38]:


master_frame.raised_amount_usd_mil.describe()


# ## Country Analysis

# In[39]:


#Spark Funds wants to invest in countries with the highest amount of funding for the chosen investment type
# Removing outliers form the data for better analysis 
master_frame = master_frame[master_frame.raised_amount_usd_mil <17600]


# In[40]:


master_frame.raised_amount_usd_mil.plot.box()


# In[41]:


master_frame = master_frame[master_frame.raised_amount_usd_mil <2000]
print(master_frame.raised_amount_usd_mil.describe())
print(master_frame.shape[0]) 
master_frame.raised_amount_usd_mil.plot.box()


# #### Only two data points are removed from the data set and now the data points are continous with less standard Deviation Using this as the new dataframe

# In[42]:


print(master_frame.groupby('country_code')['raised_amount_usd_mil'].sum().sort_values(ascending=False).head(9))
master_frame.groupby('country_code')['raised_amount_usd_mil'].sum().sort_values(ascending=False).head(9).plot.bar()


# ### From the top 9 countries USA , GBR , IND are the top 3 english speaking contries , this is decided using manual analysis
# Top English Speaking Country = USA <br/>
# Second English speaking country = GBR <br/>
# Third English speaking country = IND

# In[43]:


top9 = master_frame[master_frame['country_code'].isin(["USA","CHN","GBR","IND","CAN","FRA","ISR","DEU","JPN"])]
top9.head(3)


# ## Sector Analysis 1

# In[44]:


master_frame.category_list.value_counts().head(10)


# In[45]:


# Since some of the category_list column rows contains multiple category 
#You discuss with the CEO and come up with the business rule that the first
#string before the vertical bar will be considered the primary sector.
master_frame['primary_sector'] =master_frame['category_list'].astype(str).apply(lambda x: x.split('|')[0])
master_frame.head(3)


# In[46]:


# Analysis of mapping.csv
mapping.category_list.isnull()                     # only one category value is null 
mapping=mapping[~mapping.category_list.isnull()]   # removing the null values


# In[47]:


mapping.info()


# Observation : looking at the dataframe  , I do not see any more issues with it

# In[48]:


mapping.head()


# In[49]:


m=pd.melt(mapping, id_vars=['category_list'], var_name=['main_sector'])
m=m[m.value==1]
m=m.drop('value',axis=1)
m.shape
#print(m.sort_values('category_list'))
m


# In[50]:


m.category_list =m.category_list.replace({'0':'na'}, regex=True)
m.category_list = m.category_list.str.lower()
master_frame['primary_sector'] = master_frame['primary_sector'].str.lower()


# In[51]:


master_frame=pd.merge(master_frame,m,how="left",left_on="primary_sector",right_on="category_list")
master_frame


# In[52]:


master_frame=master_frame.drop('category_list_y',axis=1)


# In[54]:


# Creating separate Dataframe for the 3 top english venture funding type contries
D1 = master_frame[(master_frame['country_code'] == 'USA') & 
             (master_frame['raised_amount_usd_mil'] >= 5) & 
             (master_frame['raised_amount_usd_mil'] <= 15)]
D2 = master_frame[(master_frame['country_code'] == 'GBR') & 
             (master_frame['raised_amount_usd_mil'] >= 5) & 
             (master_frame['raised_amount_usd_mil'] <= 15)]
D3 = master_frame[(master_frame['country_code'] == 'IND') & 
             (master_frame['raised_amount_usd_mil'] >= 5) & 
             (master_frame['raised_amount_usd_mil'] <= 15)]


# In[55]:


print("D1 null values:" ,D1.main_sector.isnull().sum())
print("D2 null values:" ,D2.main_sector.isnull().sum())
print("D3 null values:" ,D3.main_sector.isnull().sum())


# In[56]:


# Since only one value is null we can either remove the row or we can fill using domain knowledge , but i do not have domain
# knowledge so I am removing the row instead
D1 = D1[~D1.main_sector.isnull()]


# In[57]:


print("D1 null values:" ,D1.main_sector.isnull().sum())
print("D2 null values:" ,D2.main_sector.isnull().sum())
print("D3 null values:" ,D3.main_sector.isnull().sum())


# In[58]:


D1.main_sector.value_counts()


# In[59]:


# Trying to create a count and sum using pivot table
D1.pivot_table(values = 'raised_amount_usd',index = ['main_sector'], aggfunc = {'sum','count'})


# In[60]:


#Adding total number of investment and total amount invested column to D1 dataframe
D1p=D1.pivot_table(values = 'raised_amount_usd',index = ['main_sector'], aggfunc = {'sum','count'})
D1 = D1.merge(D1p, how='left', on ='main_sector')
D1.rename(columns = {'count':'total_number'}, inplace = True)
D1.rename(columns = {'sum':'total_amount_invested'}, inplace = True)


# In[61]:


#Adding total number of investment and total amount invested column to D2 dataframe
D2p=D2.pivot_table(values = 'raised_amount_usd',index = ['main_sector'], aggfunc = {'sum','count'})
D2 = D2.merge(D1p, how='left', on ='main_sector')
D2.rename(columns = {'count':'total_number'}, inplace = True)
D2.rename(columns = {'sum':'total_amount_invested'}, inplace = True)


# In[62]:


#Adding total number of investment and total amount invested column to D3 dataframe
D3p=D3.pivot_table(values = 'raised_amount_usd',index = ['main_sector'], aggfunc = {'sum','count'})
D3 = D3.merge(D1p, how='left', on ='main_sector')
D3.rename(columns = {'count':'total_number'}, inplace = True)
D3.rename(columns = {'sum':'total_amount_invested'}, inplace = True)


# In[63]:


D1.head(3)


# In[64]:


D2.head(3)


# In[65]:


D3.head(3)


# In[66]:


D1.raised_amount_usd.count()


# In[67]:


D2.raised_amount_usd.count()


# In[68]:


D3.raised_amount_usd.count()


# ### Question:  Total number of investments (count)
# Answer :<br/> D1: 12087 <br/> D2: 622 <br/> D3: 328

# In[69]:


D1.raised_amount_usd.sum()


# In[70]:


D2.raised_amount_usd.sum()


# In[71]:


D3.raised_amount_usd.sum()


# ### Question: Total amount of investment (USD)
# Answer: <br/>
# D1: 1079570269.0 <br/>
# D2: 5394078692.0 <br/>
# D3: 2949543602.0 <br/>

# In[72]:


D1.groupby(D1.main_sector).raised_amount_usd.count().sort_values(ascending = False)


# In[73]:


D2.groupby(D2.main_sector).raised_amount_usd.count().sort_values(ascending = False)


# In[74]:


D3.groupby(D3.main_sector).raised_amount_usd.count().sort_values(ascending = False)


# ### Question: Top sector (based on count of investments)
# Answer: <br/> D1 : Others <br/> D2 : Others <br/> D3 : Others

# ### Question: Second-best sector (based on count of investments)
# Answer: <br/> D1 : Social, Finance, Analytics, Advertising <br/> D2 : Social, Finance, Analytics, Advertising <br/> D3 : Social, Finance, Analytics, Advertising

# ### Question: Third-best sector (based on count of investments)
# Answer: <br/> D1 : Cleantech / Semiconductors <br/> D2 : Cleantech / Semiconductors <br/> D3 : News, Search and Messaging

# ### Question: Number of investments in the top sector
# Answer: <br/> D1 : 2955 <br/> D2 : 148 <br/> D3 : 110

# ### Question: Number of investments in the second-best sector
# Answer: <br/> D1 : 2717 <br/> D2 : 133 <br/> D3 : 60

# ### Question: Number of investments in the third-best sector
# Answer: <br/> D1 : 2354 <br/> D2 : 130 <br/> D3 : 52

# In[75]:


# For the top sector count-wise (point 3), which company received the highest investment?
D1[D1['main_sector']=='Others'].groupby(['main_sector','permalink']).raised_amount_usd.sum().sort_values(ascending=False).head(1)


# In[76]:


D2[D2['main_sector']=='Others'].groupby(['main_sector','permalink']).raised_amount_usd.sum().sort_values(ascending=False).head(1)


# In[77]:


D3[D3['main_sector']=='Others'].groupby(['main_sector','permalink']).raised_amount_usd.sum().sort_values(ascending=False).head(1)


# ### Question: For the top sector count-wise (point 3), which company received the highest investment?
# Answer: <br/> D1: /organization/virtustream  <br/> D2: /organization/electric-cloud <br/> D3: /organization/firstcry-com 

# In[78]:


D1[D1['main_sector']=='Social, Finance, Analytics, Advertising'].groupby(['main_sector','permalink']).raised_amount_usd.sum().sort_values(ascending=False).head(1)


# In[79]:


D2[D2['main_sector']=='Social, Finance, Analytics, Advertising'].groupby(['main_sector','permalink']).raised_amount_usd.sum().sort_values(ascending=False).head(1)


# In[80]:


D3[D3['main_sector']=='Social, Finance, Analytics, Advertising'].groupby(['main_sector','permalink']).raised_amount_usd.sum().sort_values(ascending=False).head(1)


# ### Question: For the second-best sector count-wise (point 4), which company received the highest investment?
# Answer: <br/> D1: /organization/shotspotter  <br/> D2: /organization/celltick-technologies <br/> D3: /organization/manthan-systems 

# ## Checkpoint 6
# 

# In[81]:


#Making a new data frame to display the plot required in Checkpoint 6
master_frame1 = pd.merge(rounds2,companies, left_on="company_permalink", right_on="permalink", how='left')
master_frame1 = master_frame1[~(master_frame1['raised_amount_usd'].isnull())]
master_frame1 = master_frame1[~(master_frame1['country_code'].isnull())]
master_frame1 = master_frame1[~(master_frame1['category_list'].isnull())]


# In[82]:



master_frame1 = master_frame1[(master_frame1['funding_round_type'] == 'venture') 
                            | (master_frame1['funding_round_type'] == 'seed')
                            | (master_frame1['funding_round_type'] == 'angel')
                            | (master_frame1['funding_round_type'] == 'private_equity')]


# In[83]:


# increase figure size 
plt.figure(figsize=(18, 6))

# subplot 1: statistic=mean
plt.subplot(1, 3, 1)
sns.barplot(x='funding_round_type', y='raised_amount_usd', data=master_frame1)
plt.title("Average")
plt.axhline(y=5000000, linewidth=2)
plt.axhline(y=15000000, linewidth=2)

# subplot 2: statistic=median
plt.subplot(1, 3, 2)
sns.barplot(x='funding_round_type', y='raised_amount_usd', data=master_frame1, estimator=np.median)
plt.title("Median")

#subplot 3: Count of funding in each type
plt.subplot(1, 3, 3)
sns.countplot(x='funding_round_type', data=master_frame1)
plt.title("Count")


# A plot showing the fraction of total investments (globally) in angel, venture, seed, and private equity, and the average amount of investment in each funding type. This chart should make it clear that a certain funding type (FT) is best suited for Spark Funds.

# In[84]:


plt.figure(figsize=(10,5))
top9.groupby('country_code')['raised_amount_usd'].sum().sort_values(ascending=False).head(9).plot.bar()


# 
# A plot showing the top 9 countries against the total amount of investments of funding type FT. This should make the top 3 countries (Country 1, Country 2, and Country 3) very clear.

# In[85]:


plt.figure(figsize=(20, 6))
plt.subplot(1, 3, 1)
sns.barplot(y="main_sector", x="raised_amount_usd", data=D1, estimator=sum)
plt.title("Sum")
plt.show()
plt.figure(figsize=(20, 6))
plt.subplot(1, 3, 2)
sns.barplot(y="main_sector", x="raised_amount_usd", data=D2, estimator=sum)
plt.title("Sum")
plt.show()
plt.figure(figsize=(20, 6))
plt.subplot(1, 3, 3)
sns.barplot(y="main_sector", x="raised_amount_usd", data=D3, estimator=sum)
plt.title("Sum")
plt.show()


# A plot showing the number of investments in the top 3 sectors of the top 3 countries on one chart (Venture).

# In[ ]:




