import os
import numpy as np
import pydicom as dcm
import SimpleITK as sitk
from ProstateNetLoaders import SeriesPathLoaders


class SegmentationLoader:
    def __init__(self, patient_path, metadata):
        self.patient_path = patient_path
        self.metadata = metadata
        self.mask = None 
        self.OrderKey = {}
        
        ses = SeriesPathLoaders.SequenceSelectorAI(self.patient_path,self.metadata)
        ses.SetSeriesSequences(orientation="AX")
        ser_dicts = ses.GetSeriesSequences()
        for key in ser_dicts.keys():
            if ser_dicts[key] == "T2":
                ser_t2 = {key: ser_dicts[key]}
        ser = {}
        for study_item in os.listdir(self.patient_path):
            stud = os.path.join(self.patient_path, study_item)
            for key in ser_t2.keys():
                ser.update({key: os.path.join(stud, key)})
        ImObj = SeriesPathLoaders.ArrayLoad(ser)
        ImObj.LoadITKobjects()
        self.T2obj = ImObj.GetImobj()
        
    def LoadMaskPath(self):
        for stud in os.listdir(self.patient_path):
            study = os.path.join(self.patient_path,stud)
            for ser in os.listdir(study):
                if len(os.listdir(os.path.join(study,ser)))==1:
                    series = os.path.join(study,ser)
                    self.mask = dcm.dcmread(os.path.join(series,os.listdir(series)[0]))
        return self.mask
        
    def SetOrderFiles(self):
        """
        Sets a dictionary with correct axial positions. Keys are the positions in Z axis, values are an initialization of a np array
        """
        a = self.T2obj[list(self.T2obj.keys())[0]]["ReaderObj"]
        im = a.Execute()
        OrderKey = {}
        for sl in range(im.GetSize()[2]):
            im_pos = a.GetMetaData(slice = sl, key = "0020|0032")
            self.OrderKey.update({im_pos.split("\\")[2]:np.zeros((im.GetSize()[0],im.GetSize()[1]))})
    
        
    def SetPosMask(self):
        """
        sets a dictionary with keys as slice position and values the mask array 
        """
        self.dic_matc= {}
        for index,item in enumerate(self.mask[0x5200, 0x9230]):
        
            z_pos = item[0x0020, 0x9113][0][0x0020, 0x0032].value[2]
            slice_num = item[0x0020, 0x9111][0][0x0020, 0x9157].value[1]
            self.dic_matc.update({str(z_pos):self.mask.pixel_array[index]})
            
    def MatchAnno(self):
        for key in self.dic_matc.keys():
            for match_key in self.OrderKey.keys():
                if key in match_key:
                    self.OrderKey.update({match_key:self.dic_matc[key]})
        
        return np.array(list(self.OrderKey.values()))