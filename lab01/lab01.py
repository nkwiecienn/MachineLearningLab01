#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('data/housing.csv.gz')


# In[2]:


df.head()


# In[3]:


df.info()


# In[4]:


df.value_counts()


# In[5]:


df.describe()


# In[6]:


ax = df.hist(bins=50, figsize=(20,15))

fig = ax[0][0].get_figure()
fig.savefig("obraz1")


# In[7]:


ax = df.plot(kind="scatter", x="longitude", y="latitude",
    alpha=0.1, figsize=(7,4))
ax.figure.savefig("obraz2")


# In[8]:


import matplotlib.pyplot as plt
df.plot(kind="scatter", x="longitude", y="latitude",
alpha=0.4, figsize=(7,3), colorbar=True,
s=df["population"]/100, label="population",
c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig("obraz3")


# In[9]:


df.drop(columns="ocean_proximity").corr()["median_house_value"].sort_values(ascending=False)


# In[10]:


df.drop(columns="ocean_proximity").corr()["median_house_value"].sort_values(ascending=False).to_csv("korelacja.csv", index=False)


# In[11]:


import seaborn as sns
sns.pairplot(df)


# In[12]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df,
test_size=0.2,
random_state=42)
len(train_set),len(test_set)


# In[13]:


train_set.head()


# In[14]:


test_set.head()


# In[15]:


train_set.drop(columns="ocean_proximity").corr()["median_house_value"].sort_values(ascending=False)


# In[16]:


test_set.drop(columns="ocean_proximity").corr()["median_house_value"].sort_values(ascending=False)


# In[17]:


import pickle
train_set.to_pickle("trainset.pkl")
test_set.to_pickle("testset.pkl")

