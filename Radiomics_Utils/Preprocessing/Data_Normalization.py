#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler
import numpy as np
from skimage.morphology import binary_dilation,binary_opening,dilation
from scipy.ndimage import zoom
from keras_unet_collection._model_unet_2d import unet_2d
import tensorflow as tf
import cv2 as cv

class Normalization:
    """
    Creates an instance for normalization steps and takes as input an np array
    """
    def __init__(self,patients,zone):
        self.patients = patients
        self.zone = zone
        
    def norm8bit(self):
        """ Normalization of the patients as np.uint8 (Intensity scale from 0-255 integer values)
        Args:
            patients(4d np.array[Patient,slice,width,height]): patients array
        Returns:
            new_patient(4D np.array): Uint8 normalized patients (intensity scale from 0-255)
        """
        np.seterr(all="ignore")
        
        pats={}
        for key in self.patients.keys():
            arr=[]
            for j in range(self.patients[key].shape[0]):
                mn = self.patients[key][j].min()
                mx = self.patients[key].max()
                mx -= mn
                arr.append((((self.patients[key][j] - mn)/mx) * 255).astype(np.uint8))
            arr=np.asarray(arr)
            pats.update({key:arr})
        return pats
    
    def Standardization(self,min_max=False):#4d array(patients,slices,width,height)
        """ Normalization of the images patient wised
        Args:
            patients(4d np.array[Patient,slice,width,height]): patients array
            min_max(Boolean):True/False , true for min max scaler false for z-score
        Returns:
            new_patient(4D np.array): The standardized patient
        """
        np.seterr(all="ignore")
        if min_max==True:
            norm_patients={}
            for key in self.patients.keys():
                norm_slices=[]
                for slice in (self.patients[key]):
                    norm=(slice-np.min(slice))/(np.max(slice)-np.min(slice))
                    norm_slices.append(norm)
                norm_slices=np.asarray(norm_slices)
                norm_patients.update({key:norm_slices})
        else:
            scaler = StandardScaler()
            norm_patients={}
            # normalization
            for key in self.patients.keys():
                norm_slices=[]
                for slice in (self.patients[key]):
                    scaler.fit(slice.reshape(-1,1))
                    norm_slices.append(np.reshape(scaler.transform(slice.reshape(-1,1)),(slice.shape[0],slice.shape[1])))
                norm_slices=np.asarray(norm_slices)
                norm_patients.update({key:norm_slices})
        return norm_patients
    
    def bins(self,num_bins):
        """
        Args:
            self: patients
            num_bins: number of bins to partition each slice
        Returns:
            Binned patients
        """

        pats={}
        for key in self.patients.keys():
            arr=[]
            for slice in (self.patients[key]):
                bins = np.linspace(slice.min(),slice.max(), num_bins)
                arr.append(np.digitize(slice, bins))
            arr=np.asarray(arr)
            pats.update({key:arr})
        return pats
    
    def dilation_alg(self,iters):
        """ applies  dilation around the mask on annotations
        Args:
            annotations(4d np.array[Patient,slice,width,height]
            iters(int): how many times the dilation will be applied
        Returns:
            dilated_patients(4D np.arrayy): Dilated masks
        """
        dilated_patients={}
        for key in self.patients.keys():
            dilated_slices=[]
            for slice in (self.patients[key]):
                im=slice.astype(int)
                mask=dilation(im).astype(int)
                temp=mask
                for k in range(iters):
                    mask2=dilation(temp)
                    temp=mask2
                dilated_slices.append(temp)
            dilated_slices=np.asarray(dilated_slices)
            dilated_patients.update({key:dilated_slices})
        return(dilated_patients)
    
    def resize(self,x,y,anno=True):
        """ Resize the patients to certain dimensions on width and height
        Args:
            array_4d(4d np.array[Patient,slice,width,height]): patients array
            x:Width to resize
            y:Height to resize
            anno(Boolean):True/False if annotation/not (nearest neighbor if image, bilinear if annotation)
        Returns:
            processed_patients(4D np.array):resized patients
        """
        np.seterr(all="ignore")
        processed_patients={}
        if anno==False:
            for key in self.patients.keys():
                processed_slice=[]
                for slice in self.patients[key]:
                    processed_slice.append(cv.resize(slice, (x,y),cv.INTER_LINEAR))
                    #processed_slice.append(cv.convertScaleAbs(slice, alpha=alpha, beta=beta))
                processed_slice=np.asarray(processed_slice)
                processed_patients.update({key:processed_slice})  
        elif anno==True:
            for key in self.patients.keys():
                if len(self.patients[key].shape)!=2:
                    processed_slice_anno=[]
                    for slice in self.patients[key]:
                        try:
                            processed_slice_anno.append(cv.resize(slice, (x,y),cv.INTER_NEAREST))
                        except:
                            processed_slice_anno.append(zoom(slice, (x/slice.shape[0], y/slice.shape[1])))
                        #processed_slice.append(cv.convertScaleAbs(slice, alpha=alpha, beta=beta))
                    processed_slice_anno=np.asarray(processed_slice_anno)
                    processed_patients.update({key:processed_slice_anno})
                else:
                    processed_patients.update({key:self.patients[key]})
        return processed_patients 
    
    def resize_3D(self,z=25,x=256,y=256):
        """ Resize the patients to certain dimensions on width and height
        Args:
            array_4d(4d np.array[Patient,slice,width,height]): patients array
            x:Width to resize
            y:Height to resize
            anno(Boolean):True/False if annotation/not (nearest neighbor if image, bilinear if annotation)
        Returns:
            processed_patients(4D np.array):resized patients
        """
        np.seterr(all="ignore")
        processed_patients={}
        for key in self.patients.keys():
            pat = self.patients[key]
            if len(pat.shape)!=2:
                new_array = zoom(pat, (z/pat.shape[0], x/pat.shape[1], y/pat.shape[2]))
                processed_patients.update({key:new_array})
            else:
                processed_patients.update({key:pat})


        return processed_patients
    
    def Keep_area_of_interest(self,area_of_interest = True):
        ''' Returns a dictionary with keys as patients' name and values 3D numpy arrays
            in which each slice represent the part of the image indicated by the Ground
            Truth Label.
            E.g The user inputs a dictionary of T2 sequences {pat1:[[a][b][c]],pat2:[[d][e][f]]}
            and the dictonary of ground truth labels. The fucntion returns the Sequence dictionary 
            with pixels outside the indicated by the ground truth area being in 0 intensity 
            and the remaining intensities staying the same
        Args:
            self.patients,self.zone
            area_of_interest: Boolean, whether to keep the area of interest or the remaining area
        Returns:
            Dictionary: keys are the names of the patients, values are the area of interest in the initial sequence
        '''
        np.seterr(all="ignore")
        pat_gt = {}
        for key in self.zone.keys():
            arr_int = np.zeros(self.zone[key].shape)
            if self.zone[key].shape[0]>1:
                for item in range(self.zone[key].shape[0]):
                    if area_of_interest:
                        arr_int[item] = np.where(self.zone[key][item]>.5,self.patients[key][item],0)
                    else:
                        arr_int[item] = np.where(self.zone[key][item]>.5,0,self.patients[key][item])
                pat_gt.update({key:arr_int})
        return pat_gt
    
    def crop(self):
        '''
        Crops around the prostate (Utilize it after the Keep_area_of_interest function)
        self patients self zones
        
        Args:
            self: dictionaries of the patients and the corresponding tumor annotations to crop
        Returns:
            cropped : Dictionary with cropped Images on WG
            segment : Dictionary with cropped tumors  annotations on WG
        '''
        np.seterr(all="ignore")
        cropped = {}
        segment = {}
        for key in self.patients.keys():
            arr = []
            segs = []
            for ind,slice in enumerate(self.patients[key]):
                try:
                    tmp = np.where(slice!=0)
                    xmax,ymax,xmin,ymin = np.max(tmp[0]),np.max(tmp[1]),np.min(tmp[0]),np.min(tmp[1])
                    difx,dify = xmax-xmin, ymax-ymin
                    
                    if difx > dify:
                        dif = difx - dify
                        if dif/2 == dif//2:
                            ymax = ymax + dif//2
                            ymin = ymin - dif//2
                            if ymin<0:
                                ymin = 0
                        else:
                            ymax = ymax + dif//2 +1
                            ymin = ymin - dif//2
                            if ymin<0:
                                ymin = 0
                    else :
                        dif = dify - difx
                        if dif/2 == dif//2:
                            xmax = xmax + dif//2
                            xmin = xmin - dif//2
                            if xmin<0:
                                xmin = 0
                        else:
                            xmax = xmax + dif//2 +1
                            xmin = xmin - dif//2
                            if xmin<0:
                                xmin = 0       
                    arr.append(slice[xmin:xmax,ymin:ymax])
                    if self.zone[key].shape[0]>1:
                        segs.append(self.zone[key][ind,xmin:xmax,ymin:ymax])
                    else:
                        segs.append(self.zone[key])
                except:
                    arr.append(slice)
                    if self.zone[key].shape[0]>1:
                        segs.append(self.zone[key][ind])
                    else:
                        segs.append(self.zone[key])
                    
            arr,segs = np.asarray(arr),np.asarray(segs)
            cropped.update({key:arr})
            segment.update({key:segs})
        return cropped, segment
    
    def Data_cleanse(self):
        ''' Cleaning of the patients from null frames
        Args:
            patients(4D np.array,[Patients,slice,Width,Height]): Patients
            peripheral(4D np.array,[Patients,slice,Width,Height]): Annotations
        Returns:
            clear_patients(4D np.array,[Patients,slice,Width,Height]): Patients
            clear_peripheral (4D np.array,[Patients,slice,Width,Height]):Annotations
        '''
        null_pat={}
        for key in self.zone.keys():
            null_slice=[]
            for slice in range(self.zone[key].shape[0]):
                if np.sum(self.zone[key][slice])!=0:
                    null_slice.append(slice)    
            null_pat.update({key:null_slice})
        clear_patients={}
        clear_peripheral={}
        for null_key in null_pat.keys():
            clear_slices_patients=[]
            clear_slices_peripheral=[]
            for clear_slice in null_pat[null_key]:
                clear_slices_patients.append(self.patients[null_key][clear_slice])
                clear_slices_peripheral.append(self.zone[null_key][clear_slice])
            clear_slices_patients=np.asarray(clear_slices_patients)
            clear_slices_peripheral=np.asarray(clear_slices_peripheral)
            ####
            clear_patients.update({null_key:clear_slices_patients})
            clear_peripheral.update({null_key:clear_slices_peripheral})
            
        return clear_patients,clear_peripheral 




