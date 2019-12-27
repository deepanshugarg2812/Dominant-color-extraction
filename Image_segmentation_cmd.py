#!/usr/bin/env python
# coding: utf-8

# ### Dominant Color Extraction for Image Segmentation
#  - Image Segmentation
#  - Segmentation partitions an image into regions having similar visual appearance corresponding to parts of objects
#  - We will try to extract the most dominant 'K' Colors using K-Means
#  - We can apply K-Means with each pixel will reassigned to the closest of the K Colors, leading to segmentation
# 

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import cv2


# In[50]:


img = input()
img = cv2.imread(img)
plt.imshow(img)
img_ = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_)
plt.axis("off")
plt.show()


# In[8]:


### Store the size of the original image 
original_size = img_.shape


# In[13]:


### Converting the image in 2D to 1D
img_reshaped = img_.reshape((-1,3))
print(img_reshaped.shape)


# ### Using the k-means++

# In[17]:


from sklearn.cluster import KMeans


# In[28]:


k = 4
km = KMeans(n_clusters=k,init='k-means++',verbose=0)
print(type(km))


# In[29]:


km.fit(img_reshaped)


# In[33]:


centres = km.cluster_centers_
centres = np.array(centres,dtype='int8')
print(centres)


# ### Ploting of the dominant colors

# In[35]:


i = 1
plt.figure(0,figsize=(8,2))
colors = []
for each_col in centres:
    plt.subplot(1,4,i)
    plt.axis("off")
    i+=1
    colors.append(each_col)
    #Color Swatch
    a = np.zeros((100,100,3),dtype='uint8')
    a[:,:,:] = each_col
    plt.imshow(a)
plt.show()


# ### Segmenting the original image

# In[43]:


new_img = np.zeros((330*500,3),dtype='uint8')
print(new_img.shape)


# In[44]:


for ix in range(new_img.shape[0]):
    new_img[ix] = colors[km.labels_[ix]]   
new_img = new_img.reshape((original_size))
plt.imshow(new_img)
plt.axis("off")
plt.show()


# In[49]:


print('Want to save image Y/N')
read_input = input()
if read_input=='Y':
	cv2.imwrite('Segmented image.jpg',new_img)
else:
	cve.imshow('Segmented image.jpg',new_img)

# In[ ]:




