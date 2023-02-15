#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt, sklearn, os, nibabel as nib
from medpy.io import load
import SimpleITK as sitk

def Picai_load_t2(picai_path,labels_path):
    """
    Load Picai data (Patients in .mha and data in nii.gzz)
    Args:
        picai_path(str path object) : folder of the picai patients
        labels_path(str path object) : folder of the annotation path
    Returns:
        d : Dictionary of a patient's array
        l : Dictionary of a patient's label array
    """
    t2w_pats=[]
    labels=[]

    for patient in (os.listdir(picai_path)):
        for sequence in (os.listdir(os.path.join(picai_path,patient))):
            if "t2w" in sequence:
                t2w_pats.append(os.path.join(os.path.join(picai_path,patient,sequence)))

    for label in (os.listdir(labels_path)):
        labels.append(os.path.join(labels_path,label))

    # kanoume to cross match annotation data
    d,l,csPCA={},{},{}
    for l_p in labels:
        for d_p in t2w_pats:
            data_name   = os.path.basename(os.path.normpath(d_p))
            labels_name = os.path.basename(os.path.normpath(l_p))
            
            if data_name[:13]==labels_name[:13]:
                
                print("Patient - Label match",data_name[:13],"-------->",labels_name[:13])
                            
                temp1 = sitk.ReadImage(d_p)
                temp1 = sitk.DICOMOrient(temp1, 'LPS')
                temp1 = sitk.GetArrayFromImage(temp1)

                temp2 = sitk.ReadImage(l_p)
                temp2 = sitk.DICOMOrient(temp2, 'LPS')
                temp2 = sitk.GetArrayFromImage(temp2)
                d.update({data_name[:13]:temp1}),l.update({labels_name[:13]:temp2})
                
                #d.update({data_name[:13]:np.transpose(np.rot90(np.flip(load(d_p)[0], 0),k=3),axes=(2,0,1))}),l.update({labels_name[:13]:np.transpose(np.rot90(np.flip(nib.load(l_p).get_fdata(),0),k=3),axes=(2,0,1))})
    
    return d,l

def Picai_load_adc(picai_path,labels_path,register=True):
    """
    Load Picai data (Patients in .mha and data in nii.gzz)
    Args:
        picai_path(str path object) : folder of the picai patients
        labels_path(str path object) : folder of the annotation path
    Returns:
        d : Dictionary of a patient's array
        l : Dictionary of a patient's label array
    """
    adc_pats=[]
    labels=[]
    t2w_pats=[]

    for patient in (os.listdir(picai_path)):
        for sequence in (os.listdir(os.path.join(picai_path,patient))):
            if "adc" in sequence:
                adc_pats.append(os.path.join(os.path.join(picai_path,patient,sequence)))
    if register:
        for patient in (os.listdir(picai_path)):
            for sequence in (os.listdir(os.path.join(picai_path,patient))):
                if "t2w" in sequence:
                    t2w_pats.append(os.path.join(os.path.join(picai_path,patient,sequence)))

    for label in (os.listdir(labels_path)):
        labels.append(os.path.join(labels_path,label))

    # kanoume to cross match annotation data
    d,l,csPCA={},{},{}
    for l_p in labels:
        for index,d_p in enumerate(adc_pats):
            if register:
                t2_name = os.path.basename(os.path.normpath(t2w_pats[index]))
            adc_name   = os.path.basename(os.path.normpath(d_p))
            labels_name = os.path.basename(os.path.normpath(l_p))
            
            if adc_name[:13]==labels_name[:13]:
                
                print("Patient - Label match",adc_name[:13],"-------->",labels_name[:13])
                print(d_p)
                print(t2_name)     
                temp1 = sitk.ReadImage(d_p)
                temp1 = sitk.DICOMOrient(temp1, 'LPS')
                temp11 = sitk.GetArrayFromImage(temp1)
                if register:
                    tempt2 = sitk.ReadImage(t2w_pats[index])
                    tempt2 = sitk.DICOMOrient(tempt2, 'LPS')
                    tempt22 = sitk.GetArrayFromImage(tempt2)
                
                temp2 = sitk.ReadImage(l_p)
                temp2 = sitk.DICOMOrient(temp2, 'LPS')
                temp2 = sitk.GetArrayFromImage(temp2)
                if register:
                    ref,temp11 = ResampleImage_refT2(tempt2,temp1)
                
                d.update({adc_name[:13]:temp11}),l.update({labels_name[:13]:temp2})
                
                #d.update({data_name[:13]:np.transpose(np.rot90(np.flip(load(d_p)[0], 0),k=3),axes=(2,0,1))}),l.update({labels_name[:13]:np.transpose(np.rot90(np.flip(nib.load(l_p).get_fdata(),0),k=3),axes=(2,0,1))})
    
    return d,l

def Picai_load_dwi(picai_path,labels_path,register=True):
    """
    Load Picai data (Patients in .mha and data in nii.gzz)
    Args:
        picai_path(str path object) : folder of the picai patients
        labels_path(str path object) : folder of the annotation path
    Returns:
        d : Dictionary of a patient's array
        l : Dictionary of a patient's label array
    """
    t2w_pats=[]
    labels=[]
    dwi_pats=[]

    for patient in (os.listdir(picai_path)):
        for sequence in (os.listdir(os.path.join(picai_path,patient))):
            if "hbv" in sequence:
                dwi_pats.append(os.path.join(os.path.join(picai_path,patient,sequence)))
    if register:
        for patient in (os.listdir(picai_path)):
            for sequence in (os.listdir(os.path.join(picai_path,patient))):
                if "t2w" in sequence:
                    t2w_pats.append(os.path.join(os.path.join(picai_path,patient,sequence)))

    for label in (os.listdir(labels_path)):
        labels.append(os.path.join(labels_path,label))

    # kanoume to cross match annotation data
    d,l,csPCA={},{},{}
    for l_p in labels:
        for index,d_p in enumerate(dwi_pats):
            if register:
                t2_name = os.path.basename(os.path.normpath(t2w_pats[index]))
            dwi_name   = os.path.basename(os.path.normpath(d_p))
            labels_name = os.path.basename(os.path.normpath(l_p))
            
            if dwi_name[:13]==labels_name[:13]:
                
                print("Patient - Label match",dwi_name[:13],"-------->",labels_name[:13])
                            
                temp1 = sitk.ReadImage(d_p)
                temp1 = sitk.DICOMOrient(temp1, 'LPS')
                temp11 = sitk.GetArrayFromImage(temp1)
                if register:
                    tempt2 = sitk.ReadImage(t2w_pats[index])
                    tempt2 = sitk.DICOMOrient(tempt2, 'LPS')
                    tempt22 = sitk.GetArrayFromImage(tempt2)

                temp2 = sitk.ReadImage(l_p)
                temp2 = sitk.DICOMOrient(temp2, 'LPS')
                temp2 = sitk.GetArrayFromImage(temp2)
                if register:
                    ref,temp11 = ResampleImage_refT2(tempt2,temp1)
                d.update({dwi_name[:13]:temp11}),l.update({labels_name[:13]:temp2})
                
                #d.update({data_name[:13]:np.transpose(np.rot90(np.flip(load(d_p)[0], 0),k=3),axes=(2,0,1))}),l.update({labels_name[:13]:np.transpose(np.rot90(np.flip(nib.load(l_p).get_fdata(),0),k=3),axes=(2,0,1))})
    
    return d,l

def ResampleImage_refT2(reference,moving):
# Load Resample Filter
    resample = sitk.ResampleImageFilter()
# Set desired output spacing    
    #out_spacing = list((2.0, 2.0, 2.0))
    
    
#%% Resample T2 to a fixed pixel spacing    
# Get Spacing, Size and PixelIDValue(for padding) of T2    
    ref_original_spacing = reference.GetSpacing()
    ref_original_size =  reference.GetSize()



#%% Resample ADC (or DWI) to a fixed pixel spacing, origin and direction of T2      
    if moving is not None:
        # x,y,z=ref_out_size[::-1]
        xr,yr,zr = ref_original_size[::-1] #Update
        
        mov_original_spacing = moving.GetSpacing()
        mov_original_size =  moving.GetSize()
        mov_pad_value = moving.GetPixelIDValue()
        
        mov_out_size = [
                   int(np.round(
                       size * (spacing_in / spacing_out)
                   ))
                   for size, spacing_in, spacing_out in zip(mov_original_size, mov_original_spacing, ref_original_spacing)
               ]
   
        resample.SetOutputSpacing(ref_original_spacing)
        resample.SetOutputDirection(reference.GetDirection())
        resample.SetOutputOrigin(reference.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetSize(mov_out_size)
        resample.SetDefaultPixelValue(mov_pad_value)
        resample.SetInterpolator(sitk.sitkBSpline)
        mov = resample.Execute(moving)
        mov= sitk.GetArrayFromImage(mov)
        #Update
        pad_mov = np.zeros((xr,yr,zr))
        xm,ym,zm=mov.shape
        #Padding or cropping Image
        x,y,z = np.min(np.vstack(((xr,yr,zr),(xm,ym,zm))),axis=0)
        pad_mov[:x,:y,:z]=mov[:x,:y,:z]
        mov=pad_mov
        ref = sitk.GetArrayFromImage(reference)
    else:
        mov=None
#%% Return back the Arrays from itk images, along with the changes        
    return ref,mov


def labels_classification(picai_path,tum_path):
    
    """
    Load Tumor labels and converts it into Classification probelm (Patients in .mha and data in nii.gzz)
    Args:
        picai_path(str path object) : folder of the picai patients
        tum_path(str path object) : folder of the annotation path for tumors
    Returns:
        csPCA : Dictionary of a patient's labels array regarding its clinical examination value (Clinical significant Prostate cnacer or not (1/0))
        
    """
    t2w_pats=[]
    labels=[]

    for patient in (os.listdir(picai_path)):
        for sequence in (os.listdir(os.path.join(picai_path,patient))):
            if "t2w" in sequence:
                t2w_pats.append(os.path.join(os.path.join(picai_path,patient,sequence)))

    for label in (os.listdir(tum_path)):
        labels.append(os.path.join(tum_path,label))

    # kanoume to cross match annotation data
    d,l,csPCA={},{},{}
    for l_p in labels:
        for d_p in t2w_pats:
            data_name   = os.path.basename(os.path.normpath(d_p))
            labels_name = os.path.basename(os.path.normpath(l_p))
            
            if data_name[:13]==labels_name[:13]:
                
                print("Patient - Label match",data_name[:13],"-------->",labels_name[:13])
                            
                temp1 = sitk.ReadImage(d_p)
                temp1 = sitk.DICOMOrient(temp1, 'LPS')
                temp1 = sitk.GetArrayFromImage(temp1)

                temp2 = sitk.ReadImage(l_p)
                temp2 = sitk.DICOMOrient(temp2, 'LPS')
                temp2 = sitk.GetArrayFromImage(temp2)
                d.update({data_name[:13]:temp1})
                if np.sum(temp2)>0.5:
                    csPCA.update({labels_name[:13]:1})
                else:
                    csPCA.update({labels_name[:13]:0})

    return csPCA

def Enhancement_todict(test_pats,test_labs,enhanced,test_labels=None):
    """ takes dictionaries of original data and returns dictionaries of enhanced. Converts NaN values to 0
    Args:
        test_pats (dictionary): Keys are the names of each patient(ID), values are the 3D arrays for each ID
        test_labs(dictionary): Keys are the names of each patient(ID), values are the 3D annotations for each ID
        enhanced (np.array): is the enhanced 4D array (nested) 
    Returns:
        raclahe_enh (dictionary) : Keys are the names of each patient(ID), values are the 3D enhanced arrays for each ID
        labels(dictionary): Keys are the names of each patient(ID), values are the 3D annotations for each ID
    """
    
    raclahe_enh={}
    i=0
    for keys,values in test_pats.items():
        raclahe_enh.update({keys:enhanced[i]})
        i+=1

    labels={}
    i=0
    for keys,values in test_labs.items():
        try:
            labels.update({keys:test_labels[i]})
            i+=1
        except:
            labels.update({keys:values})
            i+=1
    for key,value in raclahe_enh.items():
        raclahe_enh[key][np.isnan(raclahe_enh[key])] = 0
        
    return raclahe_enh,labels

def Intensity_range(test_pats):
    """Keep the original pixel ranges with IDs in order to reconstruct the normalized image afterwards
    Args:
        test_pats (dictionary): Keys are the names of each patient(ID), values are the 3D arrays for each ID
    Returns:
        ls_ranges (list of dictionaries): each element represents a patients and each dicionary item represent the min and max intensity for all the slices sequently
    """
    ls_ranges=[]
    for patient in test_pats.values():
        original_range={"min":[],"max":[]}
        for j in range(patient.shape[0]):
            original_range["min"].append(np.min(patient[j]))
            original_range["max"].append(np.max(patient[j]))
        ls_ranges.append(original_range)
    return ls_ranges
# In[ ]:




