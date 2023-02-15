import os
import numpy as np
import pydicom as dcm
import SimpleITK as sitk

class SitkOps:
    def __init__(self, ser_path):
        self.ser_path = ser_path
        
    def LoadITKobjects(self):
        """
        Sets the reader and image sitk objects for each series into a dict
        """
        self.reader=sitk.ImageSeriesReader()
        order_files = self.reader.GetGDCMSeriesFileNames(self.ser_path)
        self.reader.SetFileNames(order_files)
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()
        _ = self.reader.Execute()
        return self.reader
    
    def PrintMetadataMatch(self) -> None:
            
        """
        Prints the dictionary with the matchings of the SITK metadata coded names
        with the real value
        """
        

        ls_keys = [key for key in self.reader.GetMetaDataKeys(slice = 0) if "ITK" not in key]
        meta_match = {}
        for key in ls_keys:
            try:
                tag_name = (key.split('|')) 
                tag_name = dcm.datadict.keyword_for_tag(tag_name)
                if not tag_name:
                    tag_name = key
            except: 
                tag_name =key
            meta_match.update({key:tag_name}) 
            
        print(meta_match)
    
    def PrintMetaValue(self, key):
        """
        Prints the given key value from a random patient
        """
        a = self.reader.GetMetaData(key = key, slice = 0)
        print(a)
        
    def to_nifti(self,mask_arr,path_save, name):
        
        Image = self.reader.Execute()
        try:
            Mask  = sitk.GetImageFromArray(mask_arr)
            Mask.SetDirection(Image.GetDirection())
            Mask.SetSpacing(Image.GetSpacing())
            Mask.SetOrigin(Image.GetOrigin())
            sitk.WriteImage(Mask, os.path.join(path_save,"Mask.nii.gz"))
        except:
            print("No mask given therefore only the series will be saved")
            sitk.WriteImage(Image, os.path.join(path_save,"{}.nii.gz".format(name)))