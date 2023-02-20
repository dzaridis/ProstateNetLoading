import os
import numpy as np
import pydicom as dcm
import SimpleITK as sitk
import ProstateNetLoaders.LookUpTable
from ProstateNetLoaders import SitkOperations

class ProstateNetPathLoaders:
    def __init__(self, patient_path) -> None:
        
        self.patient_path = patient_path
        self.ser = {}
        self.ser_objects = {}
        self.ser_descr = {}
        
    def SeriesLoader(self) -> None:
        """
        Loads the series path in each file

        Returns:
            dict: keys are the names of each patient series while value is the description of the series, whether to be T2 ADC etc. 
        """
        self.ser = {}
        for study_item in os.listdir(self.patient_path):
            stud = os.path.join(self.patient_path, study_item)
            for series_item in os.listdir(stud):
                if os.listdir(os.path.join(stud, series_item))!= 1:
                    self.ser.update({series_item: os.path.join(stud, series_item)}) 
                
    def LoadObjects(self) -> None:
        """
        Sets the reader and image sitk objects for each series into a dict
        """
        self.ser_objects = {}
        for key in self.ser.keys():
            reader = SitkOperations.SitkOps(self.ser[key]).LoadITKobjects()
            self.ser_objects.update({key: {"Reader Obj": reader}})
    
    def LoadSeriesDescription(self) -> None:
        """
        Loads the dictionary with series name and descriptions
        """
        for key in self.ser_objects.keys():
            self.ser_descr.update({key: self.ser_objects[key]["Reader Obj"].GetMetaData(key = "0008|103e", slice = 0).strip()})

    
    def GetSitkObjSerDescr(self):
        """
        Returns a dictionary with series name as keys and series path as value for each key

        Returns:
            dict: keys are the names of each patient series while value is the path to the corresponding series
            dict: keys are the names of each patient series while value is the description
        """
        return self.ser , self.ser_descr
    
class SequenceSelectorHeuristics():
    def __init__(self, patient_path):
        """Constructor for the sequence selector. Preferably utilized
           after the ProstateNetLoaders.SeriesPathsLoaders.ProstateNetPathLoaders Class
           in order to obtain  ser_description and series_paths

        Args:
            str file path: patient's path 
        """
        self.patient_path = patient_path
        ImObj = ProstateNetPathLoaders(self.patient_path)
        ImObj.SeriesLoader()
        ImObj.LoadObjects()
        ImObj.LoadSeriesDescription()
        self.serobj, self.desc = ImObj.GetSitkObjSerDescr() # returns the description of each series in a dict for a single patient
        
            
    def PrintAvailableSequenceFromLookUp(self):
        """
        Prints the available sequences from the Heuristic dictionary
        """
        lookup = ProstateNetLoaders.LookUpTable.GetLookUpTable()
        print("Available keys to select from are the following:")
        for key in lookup.keys():
            print(key)
    
    def SetSeriesSequences(self,orientation = "AX"):
        """
        Sets the dictionary with keys as the series names and values as the corresponding Series Description in a homogenized form
        Args
            str : orientation. Permitted values "AX", "SAG", "COR" 
        """
        self.NormDesc = {}
        for key,value in self.desc.items():
            for keylup,valuelup in ProstateNetLoaders.LookUpTable.GetLookUpTable().items():
                if value in valuelup:
                    try:
                        if orientation == "AX":
                            if "TRA" in value.upper() or "AX" in value.upper():
                                self.NormDesc.update({key: keylup})
                        elif orientation == "SAG":
                            if "SAG" in value.upper():
                                self.NormDesc.update({key: keylup})
                        elif orientation == "COR":
                            if "COR" in value.upper():
                                self.NormDesc.update({key: keylup})
                    except:
                        self.NormDesc.update({key: keylup})
        
    def GetSeriesSequences(self):
        """Returns the dict of series sequence

        Returns:
            dict: keys are the series names, values are the homogenized sequence description name ("T2" or "ADC", or "DWI)
        """
        return self.NormDesc
    
class SequenceSelectorAI():
    def __init__(self, patient_path, metadata):
        """Constructor for the sequence selector. Preferably utilized
           after the ProstateNetLoaders.SeriesPathsLoaders.ProstateNetPathLoaders Class
           in order to obtain  ser_description and series_paths

        Args:
            str file path: patient's path 
            csv file: contains information regarding series description information from Jose's Docker 
        """
        self.patient_path = patient_path
        self.metadata = metadata
        ImObj = ProstateNetLoaders.SeriesPathLoaders.ProstateNetPathLoaders(self.patient_path)
        ImObj.SeriesLoader()
        ImObj.LoadObjects()
        ImObj.LoadSeriesDescription()
        self.serobj, self.desc = ImObj.GetSitkObjSerDescr() # returns the description of each series in a dict for a single patient
        
            
    
    def SetSeriesSequences(self,orientation = "AX", Heuristics = False):
        """
        Sets the dictionary with keys as the series names and values as the corresponding Series Description in a homogenized form
        Args
            str : orientation. Permitted values "AX", "SAG", "COR" 
        """
        self.NormDesc = {}
        for key,value in self.desc.items():
            
            if Heuristics:
                a = self.metadata[self.metadata["series_uid"] == key]["series_type_heuristics"].values
            else:
                a = self.metadata[self.metadata["series_uid"] == key]["series_type"].values
            if orientation == "AX":
                if "tra" in value or "TRA" in value or "AX" in value or "Axial" in value or "AXIAL" in value or "axial" in value or "ax" in value or "Ax" in value or "Tra" in value:
                    self.NormDesc.update({key: a})
            elif orientation == "SAG":
                if "sag" in value or "SAG":
                    self.NormDesc.update({key: a})
            elif orientation == "COR":
                if "cor" in value or "COR":
                    self.NormDesc.update({key: a})
            else:
                self.NormDesc.update({key: a})

        
    def GetSeriesSequences(self):
        """Returns the dict of series sequence

        Returns:
            dict: keys are the series names, values are the homogenized sequence description name ("T2" or "ADC", or "DWI)
        """
        return self.NormDesc    



class ArrayLoad:
    def __init__(self, ser_dc):
        """Constructor of ArrayLoad class

        Args:
            ser_dc (dict): keys are the series name, values are the series path
        """
        self.ser_dc = ser_dc
        self.ar = {}
        self.obj = {}
    
    def LoadITKobjects(self):
        
        """
        Sets the reader and image sitk objects for each series into a dict
        """
        
        key = list(self.ser_dc.keys())[0]
        
        ITKread = SitkOperations.SitkOps(self.ser_dc[key]).LoadITKobjects()
        ITKimage = ITKread.Execute()
        self.obj.update({key: {"ReaderObj": ITKread,"ImageObj": ITKimage}})
        self.ar.update({key: sitk.GetArrayFromImage(ITKimage)})
    
    def GetArray(self):
        """Returns the dictionary containing the image in numpy array format

        Returns:
            dict: keys are the series names, values are the numpy arrays
        """
        return self.ar
    
    def GetImobj(self):
        """Returns the dictionary containing the image as Sitk object

        Returns:
            dict: keys are the series names, values are the Sitk objects
        """
        return self.obj
