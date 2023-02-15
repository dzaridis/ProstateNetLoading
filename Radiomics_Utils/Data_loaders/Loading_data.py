#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from glob import glob
import nibabel as nib
def data(path_images):

    data=[]
    for index,files in enumerate(glob(path_images+"\*.nii.gz")):
        print(files)
        All_data=nib.load(files)
        img_data=(np.array(All_data.get_fdata()))
        data.append(img_data)
    return(data)
def labels(path_labels):
    labels=[]
    for index,files in enumerate(glob(path_labels+"\*.nii.gz")):
        All_data=nib.load(files)
        img_data=(np.array(All_data.get_fdata()))
        labels.append(img_data)
    return(labels)

def unified(d,l,slices):# slices na kratisei tous astheneis me sugkekrimeno slice
    data_uni=[]
    labels_uni=[]
    for i in range (len(d)):
        if (d[i].shape[2]==slices):
            data_uni.append(d[i])
    for i in range (len(l)):   
        if (l[i].shape[2]==slices):
            labels_uni.append(l[i])
    data_uni=np.array(data_uni)
    labels_uni=np.array(labels_uni)
    return([data_uni,labels_uni])


# In[ ]:




