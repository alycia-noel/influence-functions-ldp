import numpy as np
import pandas as pd
import fairlearn 

def get_diabetes():
    #loading Dataset
    data = fairlearn.datasets.fetch_diabetes_hospital().data
    data = data.drop(['readmitted'], axis = 1)
    data = data[data.race != 'Unknown'] 

    data = data[data.gender != 'Unknown/Invalid'] 
    
    data['change'] = data['change'].replace('Ch', 1)
    data['change'] = data['change'].replace('No', 0)
    
    data['gender'] = data['gender'].replace('Male', 1)
    data['gender'] = data['gender'].replace('Female', 0)
    
    data['diabetesMed'] = data['diabetesMed'].replace('Yes', 1)
    data['diabetesMed'] = data['diabetesMed'].replace('No', 0)


    data['insulin'] = data['insulin'].replace('No', 0)
    data['insulin'] = data['insulin'].replace('Steady', 1)
    data['insulin'] = data['insulin'].replace('Up', 1)
    data['insulin'] = data['insulin'].replace('Down', 1)

    data['A1Cresult'] = data['A1Cresult'].replace('>7', 1)
    data['A1Cresult'] = data['A1Cresult'].replace('>8', 1)
    data['A1Cresult'] = data['A1Cresult'].replace('Norm', 0)
    data['A1Cresult'] = data['A1Cresult'].replace('None', -1)
    
    data['max_glu_serum'] = data['max_glu_serum'].replace('>200', 1)
    data['max_glu_serum'] = data['max_glu_serum'].replace('>300', 1)
    data['max_glu_serum'] = data['max_glu_serum'].replace('Norm', 0)
    data['max_glu_serum'] = data['max_glu_serum'].replace('None', -1)

    data['medicare'] = data['medicare'].replace('True', 1)
    data['medicare'] = data['medicare'].replace('False', 0)
    
    data['medicaid'] = data['medicaid'].replace('True', 1)
    data['medicaid'] = data['medicaid'].replace('False',0)
    
    data['had_emergency'] = data['had_emergency'].replace('True', 1)
    data['had_emergency'] = data['had_emergency'].replace('False',0)
    
    data['had_inpatient_days'] = data['had_inpatient_days'].replace('True', 1)
    data['had_inpatient_days'] = data['had_inpatient_days'].replace('False',0)
    
    data['had_outpatient_days'] = data['had_outpatient_days'].replace('True', 1)
    data['had_outpatient_days'] = data['had_outpatient_days'].replace('False',0)
    
    data['num_lab_procedures'] = data['num_lab_procedures'].apply(lambda x: round(x/10))
    data['num_medications'] = data['num_medications'].apply(lambda x: round(x/10))
    
    
    data = pd.get_dummies(data, columns=['age', 'race', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'medical_specialty','num_lab_procedures', 'num_procedures', 'num_medications', 'primary_diagnosis','number_diagnoses'], drop_first = False)
  
    return data