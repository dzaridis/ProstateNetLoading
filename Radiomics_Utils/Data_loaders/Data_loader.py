#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import pydicom as dcm
import csv
from sklearn.decomposition import PCA


# In[2]:


def Data_load(metadata_path,patients_path):
    """ Loads the Prostate X1 dataset (98 patients)
    Args:
        metadata_path(str path): Path to the metadata file in order to correctly load the data
        patients_path(str path): Path to the folder where all the patients are located

    Returns:
        patients_rev(4D np.array,[Patients,slice,Width,Height]):Patients
    """

    metadata=pd.read_csv(metadata_path)
    mr_modality=metadata[metadata["Modality"]=="MR"]
    mr_modality  = mr_modality.sort_values(by="Subject ID")

    patients=[]
    patients_rev=[]
    for file_path in mr_modality["File Location"]:
        #print(mr_modality["Subject ID"])
        slices=[]
        slices_rev=[]
        for index,dicom_file in enumerate(glob(file_path+"/*")):
            slices.append(dcm.dcmread(dicom_file).pixel_array)
            #print(dicom_file)
        #slices_rev=slices.reverse()  
        slices=np.asarray(slices)
        slices_rev=np.flipud(slices)

        #print("The shape of a patients scan is:",slices.shape,slices_rev.shape)
        patients.append(slices)
        patients_rev.append(slices_rev)
    print("Data succesfully loaded (check for reverse loading)")
    patients=np.asarray(patients)
    patients_rev=np.asarray(patients_rev)
    return(patients_rev)


# In[3]:


def Labels_load(metadata_path,segmentation_path):
    """ Loads the Prostate X1 dataset's annotation (98 patients' masks)
    Args:
        metadata_path(str path): Path to the metadata file in order to correctly load the data
        segmentation_path(str path): Path to the folder where all the patients' segmentations are located

    Returns:
        patient_peripheral(4D np.array,[Patients,slice,Width,Height]):Peripheral zone
        patient_transition (4D np.array,[Patients,slice,Width,Height]):Transition zone
        patient_urethra(4D np.array,[Patients,slice,Width,Height]):Urethra
        patient_anterior(4D np.array,[Patients,slice,Width,Height]):Anterior fibromuscular   
    """
    metadata=pd.read_csv(metadata_path)
    seg_modality=metadata[metadata["Modality"]=="SEG"]
    seg_modality = seg_modality.sort_values(by="Subject ID")
    patient_peripheral=[]
    patient_transition=[]
    patient_urethra=[]
    patient_anterior=[]
    patients_lab=[]
    for file_path in seg_modality["File Location"]:
        #print(seg_modality["Subject ID"].iterows)
        slices_lab=[]
        for index,dicom_file in enumerate(glob(file_path+"/*")):
            slices_lab.append(dcm.dcmread(dicom_file))
            #print(dicom_file)
        slices_lab=np.asarray(slices_lab)
        patients_lab.append(slices_lab)
    patients_lab=np.asarray(patients_lab)
    cnt=0
    for patient_label in ((patients_lab)):
        #image=patient_label.pixel_array
        print("patient and its images arrays :",cnt)
        cnt+=1
        peripheral_zone=[]
        transition_zone=[]
        urethra=[]
        anterior_fibromuscular=[]
        an_index= patient_label[0].pixel_array.shape[0]//4
        print("patient has ",an_index," slices")
        for ind,image in enumerate(patient_label[0].pixel_array):
            if ind<an_index:
                peripheral_zone.append(image)
            elif ind<(2*an_index) and ind>=(an_index):
                transition_zone.append(image)
            elif ind<(3*an_index) and ind>=(2*an_index): 
                urethra.append(image)
            elif ind<(4*an_index) and ind>=(3*an_index): 
                anterior_fibromuscular.append(image)
        print(len(peripheral_zone),len(transition_zone),len(urethra),len(anterior_fibromuscular))
        peripheral_zone=np.asarray(peripheral_zone)
        transition_zone=np.asarray(transition_zone)
        urethra=np.asarray(urethra)
        anterior_fibromuscular=np.asarray(anterior_fibromuscular)
        patient_peripheral.append(peripheral_zone)
        patient_transition.append(transition_zone)
        patient_urethra.append(urethra)
        patient_anterior.append(anterior_fibromuscular)
    print("Peripheral, Transition, Urethra and anterior Fibromuscular segmentations succesfully loaded")
    patient_peripheral=np.asarray(patient_peripheral)
    patient_transition=np.asarray(patient_transition)
    patient_urethra=np.asarray(patient_urethra)
    patient_anterior=np.asarray(patient_anterior)
    return(patient_peripheral, patient_transition,patient_urethra,patient_anterior)
    
def data_parse(patients_train,per_train,patients_val,per_val,patients_test,per_test):
    ''' Flattens the 4D numpy array train,validation and test patients and corresponding annotations into 3D numpy arrays to prepare them for training 
    Args:
        patients_train(4D np.array,[Patients,slice,Width,Height]): Train Patients
        per_train(4D np.array,[Patients,slice,Width,Height]): Train Labels
        patients_val(4D np.array,[Patients,slice,Width,Height]): Validation Patients
        per_val(4D np.array,[Patients,slice,Width,Height]): Validation Labels
        patients_test(4D np.array,[Patients,slice,Width,Height]): Test Patients
        per_test(4D np.array,[Patients,slice,Width,Height]): Test Labels

    Returns:
        data_train(3D np.array,[slice,Width,Height]): Train slices
        labels_train (3D np.array,[slice,Width,Height]):Train Labels
        data_val(3D np.array,[slice,Width,Height]):Validation slices
        labels_val(3D np.array,[slice,Width,Height]):Validation labels
        data_test(3D np.array,[slice,Width,Height]):Test slices
        labels_test(3D np.array,[slice,Width,Height]):Test labels
    '''
    data_train,labels_train=[],[]
    for patient,label in zip(patients_train,per_train):
        for pat,lab in zip(patient,label):
            data_train.append(pat)
            labels_train.append(lab)
    data_train = np.asarray(data_train)
    labels_train = np.asarray(labels_train)
    print("data train shape is: ",data_train.shape,"labels train shape is: ",labels_train.shape)
    
    data_val,labels_val=[],[]
    for patient,label in zip(patients_val,per_val):
        for pat,lab in zip(patient,label):
            data_val.append(pat)
            labels_val.append(lab)
    data_val = np.asarray(data_val)
    labels_val = np.asarray(labels_val)
    print("data validation shape is: ",data_val.shape,"labels validation shape is: ",labels_val.shape)
    
    
    data_test,labels_test=[],[]
    for patient,label in zip(patients_test,per_test):
        for pat,lab in zip(patient,label):
            data_test.append(pat)
            labels_test.append(lab)
    data_test = np.asarray(data_test)
    labels_test = np.asarray(labels_test)
    print("data test shape is: ",data_test.shape,"labels test shape is: ",labels_test.shape)
    return(data_train,labels_train,data_val,labels_val,data_test,labels_test)

def bin_selection(dataframe,value_low,value_high):
    df=dataframe[dataframe["Proportion of foreground pixel in the patient"]>value_low]
    df=df[df["Proportion of foreground pixel in the patient"],value_high]
    return df
 
def Pixel_counting(label):
    """
    Takes the binary mask as an input and counts the white and black pixels
    """
    Patient_statistics = pd.DataFrame()
    bg_pixels=[]
    fg_pixels=[]
    prp_all=[]
    prp_rel=[]
    index_arr=[]
    for index,patient in enumerate(label):
        background_pixels = np.unique(patient,return_counts=True)[1][0]
        foreground_pixels = np.unique(patient,return_counts=True)[1][1]
        proportion_all = foreground_pixels/(background_pixels+foreground_pixels)
        proportion_rel = background_pixels/foreground_pixels
        bg_pixels.append(background_pixels)
        fg_pixels.append(foreground_pixels)
        prp_all.append(proportion_all*100)
        prp_rel.append(proportion_rel)
        index_arr.append(index)
    Patient_statistics["Patient"]= index_arr
    Patient_statistics["Background Pixels"]= bg_pixels
    Patient_statistics["Foreground Pixels"] = fg_pixels
    Patient_statistics["Proportion of foreground pixel in the patient %"] = prp_all
    Patient_statistics["Background Pixels/Foreground Pixels in the patient"] = prp_rel
    
    return(Patient_statistics)


def PCA_apply(clear_data,n_comp,norm=True):
    """
    args:
    -3D array [slices,X,Y],dtype:float/int
    -Number of components to calculate,dtype:integer
    -Normalization of 3D data, dtype:boolean
    returns:
    -2D array of components [Num_samples,Number of components]
    -2D array of explained variance ratio
    """
    if norm:
        norm_slices=[]
        for slice in (clear_data):
            norm=(slice-np.min(slice))/(np.max(slice)-np.min(slice))
            norm_slices.append(norm)
        clear_data=np.asarray(norm_slices)
    
    
    clear_flat=np.zeros((clear_data.shape[0],clear_data.shape[1]*clear_data.shape[2]))
    for i in range(clear_data.shape[0]):
        clear_flat[i]=clear_data[i].flatten()
    print(clear_flat.shape)
    pca = PCA(n_comp)
    comp=pca.fit_transform(clear_flat)
    exp_var_rat=pca.fit(clear_flat).explained_variance_ratio_
    return(comp,exp_var_rat)

def PCA_Diagramms(clear_dataA,clear_dataB,n_comp,NameA,NameB,s_path,norm=True,save=True):
    """
    Args:
    -clear_dataA:3D array [slices,X,Y] Dataset A,dtype:float/int
    -clear_dataB:3D array [slices,X,Y] Dataset B,dtype:float/int
    -n_comp: Number of components to compute
    -NameA:DatasetA name , str
    -NameB:datasetB name , str
    -s_path:Give the Save path
    -save:Boolean whether to save the figures
    Returns:
    -Diagramm objects
    """
    CompA,ExpA=PCA_apply(clear_dataA,n_comp,norm=True)
    CompB,ExpB=PCA_apply(clear_dataB,n_comp,norm=True)
    
    df_A=pd.DataFrame({"First Component":CompA[:,0],"Dataset":NameA})
    df_B=pd.DataFrame({"First Component":CompB[:,0],"Dataset":NameB})
    for i in range(CompA.shape[1]):
        df_A[str(i+1)+" Component"]=CompA[:,i]
    for i in range(CompB.shape[1]):
        df_B[str(i+1)+" Component"]=CompB[:,i]
    total=pd.concat([df_A,df_B]) 
    cols=total.columns.drop(["First Component","Dataset"])
    print(cols)
    fig, axes = plt.subplots( n_comp//5, 5, figsize=(20,20),sharey=True)
    ax=axes.flatten()
    for index,name in enumerate(cols): 
        sns.kdeplot(ax=ax[index],data=total,x=name,hue="Dataset",shade="true")
        #ax[index].set_title("Gaussian Kernel density estimation of "+name,size=12)
    if save==True:
        fig.savefig(s_path+"//Components.png",bbox_inches='tight',dpi=400)
    plt.figure()    
    cdf_A = np.cumsum(ExpA)
    cdf_B = np.cumsum(ExpB)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False, 'figure.facecolor': 'white',}
    sns.set(font_scale = 1.5)
    sns.set_theme(rc=custom_params)
    sns.set_context("paper")
    sns.lineplot(data=cdf_A)
    sns.lineplot(data=cdf_B)
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.legend(["Prostate3T","ProstateX"])
    if save==True:
        plt.savefig(s_path+"//Explained_Variance.png",bbox_inches='tight',dpi=400)
    return(fig,plt)


# In[ ]:





# In[ ]:




