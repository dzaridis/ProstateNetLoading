import os, nibabel as nib, numpy as np


def P158tT2_Load(pat_path):
    for ser in os.listdir(pat_path):
        if ser == "t2.nii.gz":
            img = nib.load(os.path.join(pat_path,ser)).get_fdata()
            img = np.transpose(img,(2,1,0)) 
    return img 

def P158ADC_Load(pat_path):
    for ser in os.listdir(pat_path):
        if ser == "adc.nii.gz":
            img = nib.load(os.path.join(pat_path,ser)).get_fdata()
            img = np.transpose(img,(2,1,0)) 
    return img 

def P158DWI_Load(pat_path):
    for ser in os.listdir(pat_path):
        if ser == "dwi.nii.gz":
            img = nib.load(os.path.join(pat_path,ser)).get_fdata()
            img = np.transpose(img,(2,1,0)) 
    return img 

def P158_anatomies_Load(pat_path):
    for ser in os.listdir(pat_path):
        if "anatomy" in ser:
            img = nib.load(os.path.join(pat_path,ser)).get_fdata()
            img = np.transpose(img,(2,1,0))
            per = np.where(img==2., 1, 0).astype(int)
            tra = np.where(img==1., 1, 0).astype(int)
            wg  = np.where(img==0., 0, 1).astype(int)
    return wg, per, tra

def P158_adc_tumor_Load(pat_path):
    for ser in os.listdir(pat_path):
        if ser == "adc_tumor_reader1.nii.gz":
            img = nib.load(os.path.join(pat_path,ser)).get_fdata()
            img = np.transpose(img,(2,1,0))
            img = np.where(img>.9, 1, 0).astype(int)
    return img
def P158_t2_tumor_Load(pat_path):
    for ser in os.listdir(pat_path):
        if "t2_tumor" in ser:
            img = nib.load(os.path.join(pat_path,ser)).get_fdata()
            img = np.transpose(img,(2,1,0))
            img = np.where(img>.9, 1, 0).astype(int)
    return img

def Prostate158(p158_path):
    """
    Args:
        path file, a folder all_pats containing all the patients
    Returns:
        Patients' dictionary, whole gland dictionary-annotation, transition dictionary-annotations, peripheral dictionary-annotations
    """
    T2_158 = {}
    ADC_158 = {}
    DWI_158 = {}
    Per_158 = {}
    Tra_158 = {}
    WG_158 = {}
    Tum_adc_158 = {}
    Tum_t2_158 = {}
    for patient in os.listdir(p158_path):
        pat = os.path.join(p158_path,patient)
        T2_158.update({patient:P158tT2_Load(pat)})
        ADC_158.update({patient:P158ADC_Load(pat)})
        DWI_158.update({patient:P158DWI_Load(pat)})
        wg, per, trans= P158_anatomies_Load(pat)
        WG_158.update({patient:wg})
        Per_158.update({patient:per})
        Tra_158.update({patient:trans})
        try:
            tums_adc = P158_adc_tumor_Load(pat)
        except:
            tums_adc = np.array([0])
        try:
            tums_t2 = P158_t2_tumor_Load(pat)
        except:
            tums_t2 = np.array([0])
            
        Tum_adc_158.update({patient:tums_adc})
        Tum_t2_158.update({patient:tums_t2})
    return T2_158,ADC_158,DWI_158, WG_158, Tra_158, Per_158, Tum_adc_158, Tum_t2_158