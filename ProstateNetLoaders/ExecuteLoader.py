from ProstateNetLoaders import *

class Execute:
    def __init__(self, patient_path, metadata):
        self.patient_path = patient_path
        self.metadata = metadata
        self.arr =None
        self.anno=None
    def LoadArrays(self, orientation = "AX", seq = "T2"):
        """_summary_

        Args:
            orientation (str, optional): plane to select from e.g "AX", "COR", "SEG". Defaults to "AX".
            seq (str, optional): Sequence to select from. Available sequences are : 'DCE', 'OTHERS', 'DWI', 'T2', 'ADC'. Defaults to "T2".
        """
        ImObj = ProstateNetLoaders.SeriesPathLoaders.ProstateNetPathLoaders(self.patient_path )
        ImObj.SeriesLoader() # loads the series dictionaries ({series name : series path})
        ImObj.LoadObjects() # create sitk objects to extract information from
        ImObj.LoadSeriesDescription() # returns the description of each series in a dict for a single patient
        ser, descr = ImObj.GetSitkObjSerDescr() 
        ses = ProstateNetLoaders.SeriesPathLoaders.SequenceSelectorAI(self.patient_path, self.metadata)
        ses.SetSeriesSequences(orientation=orientation)
        ser_dicts = ses.GetSeriesSequences()
        self.seq = {}
        for key in ser_dicts.keys():
            if ser_dicts[key] == seq:
                self.seq.update({key:ser[key]})
        
        arrobj = ProstateNetLoaders.SeriesPathLoaders.ArrayLoad(self.seq)
        arrobj.LoadITKobjects()
        self.arr =arrobj.GetArray()
        im = arrobj.GetImobj()
        
        ld = ProstateNetLoaders.SegmentationLoaders.SegmentationLoader(self.patient_path, self.metadata)
        ms  = ld.LoadMaskPath()
        ld.SetOrderFiles()
        ld.SetPosMask()
        self.Anno = ld.MatchAnno()
    
    def GetItems(self):
        return self.arr, {list(self.arr.keys())[0]:self.Anno}