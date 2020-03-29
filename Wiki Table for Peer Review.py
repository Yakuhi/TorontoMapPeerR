#!/usr/bin/env python
# coding: utf-8

# ## Import needed libraries

# In[1]:


import pandas as pd
import numpy as np
import requests


# ### Load table into database

# In[2]:


df = pd.read_html('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M')[0]


# In[3]:


for col in df.columns: 
    print(col) 


# In[4]:


df.columns = ['Row_1', 'Row_2', 'Row_3', 'Row_4','Row_5', 'Row_6', 'Row_7', 'Row_8','Row_9']


# In[5]:


x = df['Row_1'].unique().tolist()
x1 = df['Row_2'].unique().tolist()
x2 = df['Row_3'].unique().tolist()
x3 = df['Row_4'].unique().tolist()
x4 = df['Row_5'].unique().tolist()
x5 = df['Row_6'].unique().tolist()
x6 = df['Row_7'].unique().tolist()
x7 = df['Row_8'].unique().tolist()
x8 = df['Row_9'].unique().tolist()


# Creating a new list of values to make a new Data frame to work with

# In[6]:


z = x + x1 +x2 +x3 +x4 +x5 +x6 +x7 +x8
new_df = pd.DataFrame(z)

for col in new_df.columns: 
    print(col) 
    
new_df.columns = ['Locate']


# Modifying string data to extract needed column values

# In[7]:


new_df['Locate'] = new_df[~new_df.Locate.str.contains("Not assigned", na=False)]
new_df = new_df.dropna()

new_df['Postal_Code'] = new_df['Locate'].str[:3]

new_df['Borough'] = new_df['Locate'].str[3:]

new_df['Borough']= new_df['Borough'].str.split("(", n = 1, expand = True)

new_df['Neighborhood'] = new_df['Locate'].str[3:]


# In[8]:


new = new_df['Neighborhood'].str.split("(", n = 1, expand = True) 

new_df['Borough']= new[0] 

new_df['Neighborhood']= new[1] 


# Clean up

# In[ ]:


new_df.drop(columns =["Locate"], inplace = True) 
new_df['Neighborhood'] = new_df['Neighborhood'].str.replace(')',' ')
new_df['Neighborhood'] = new_df['Neighborhood'].str.replace('/',', ')
new_df.reset_index()


# In[61]:


new_df.shape


# In[59]:


new_df.head(20)


# ## Finding the coordinates to go with postal codes

# In[53]:


df1 = pd.read_csv('https://cocl.us/Geospatial_data')


# In[67]:


df1.columns = ['Postal_Code','Latitude','Longitude']


# In[70]:


New_df = pd.merge(new_df,
                 df1[['Postal_Code','Latitude','Longitude']],
                 on='Postal_Code')


# Table Now with GPS Cooordinates

# In[71]:


New_df


# # Make Clusters of different nighborhoods in Toronto

# In[95]:


import random 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import folium
print('Libraries imported.')


# In[ ]:





# In[76]:


dfk = New_df.drop(['Postal_Code','Borough','Neighborhood'], axis = 1)
dfk.head()


# In[ ]:


X = dfk.values[:,1:]
X = np.nan_to_num(X)
cluster_dataset = StandardScaler().fit_transform(X)
cluster_dataset


# In[80]:


num_clusters = 3

k_means = KMeans(init="k-means++", n_clusters=num_clusters, n_init=12)
k_means.fit(cluster_dataset)
labels = k_means.labels_

print(labels)


# In[83]:


New_df["Labels"] = labels
New_df.head(20)


# ## Will create points to show centre of clusters

# In[144]:


New_df1 = New_df.groupby('Labels').mean()


# In[151]:


New_df1['Label'] = {'East', 'Center', 'West'}


# In[86]:


k_means_labels = k_means.labels_
k_means_labels


# In[88]:


k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers


# ### Plotting the map

# In[165]:



map_toronto = folium.Map(location=[43.746157, -79.265587], zoom_start=10)


for lat, lng, borough, neighborhood, borough1  in zip(New_df['Latitude'], New_df['Longitude'], New_df['Labels'], New_df['Neighborhood'], New_df['Borough']):
    label = '{}, {}'.format(neighborhood, borough1)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=10,
        popup=label,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity= (0.1 + borough/2),
        parse_html=False).add_to(map_toronto)  

## Three markers to identify three groups zones    

for lat, lng, labe in zip(New_df1['Latitude'], New_df1['Longitude'], New_df1['Label']):
    label = '{}'.format( labe)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=60,
        popup=label,
        color=None,
        fill=True,
        fill_color='red',
        fill_opacity= 0.2,
        parse_html=False).add_to(map_toronto)      
    
map_toronto


# The map display each neighborhood, with color shades to indicate those belonging to similar groupings. 
# The red circles indicate the centre of each group.

# In[ ]:




