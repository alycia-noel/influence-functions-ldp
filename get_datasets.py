import numpy as np
import pandas as pd
import fairlearn 

def get_diabetes():
    #loading Dataset
    data = fairlearn.datasets.fetch_diabetes_hospital().data
    data = data.drop(['readmitted'], axis = 1)
    data = data[data.race != 'Unknown'] 
    #data = data[data.max_glu_serum != 'None']
    #data = data[data.A1Cresult != 'None']
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


def get_adult():
    features = ['age', 'workclass', 'final_weight', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_class']

    data = pd.read_csv('data/adult.data', comment='|', skipinitialspace=True, names=features,
            na_values={
                'capital_gain': 99999,
                'workclass': '?',
                'native_country': '?',
                'occupation': '?'
            },
        )
    
    data = data.replace({'<=50K.' : '<=50K', '>50K.' : '>50K'})

    data = (data[(data['workclass'] != '?') & (data['occupation'] != '?') & (data['workclass'] != 'Without-pay')].reset_index(drop=True))
    data.loc[data['workclass'].isin(['State-gov', 'Federal-gov', 'Local-gov']), 'workclass'] = 'Government'
    data.loc[data['workclass'].isin(['Self-emp-not-inc', 'Self-emp-inc']), 'workclass'] = 'Self-Employed'
    data.loc[data['workclass'].isin(['Private']), 'workclass'] = 'Privately-Employed'

    data.loc[data['occupation'] == 'Armed-Forces', 'occupation'] = 'Protective-serv'

    data.loc[data['marital_status'].isin(['Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent']), 'marital_status'] ='Married'
    data.loc[data['marital_status'].isin(['Divorced', 'Never-married', 'Separated', 'Widowed']), 'marital_status'] = 'Not-married'

    data.loc[data['education_num'] <= 8, 'education'] = 'Less than High School'
    data.loc[data['education_num'].isin([9, 10]), 'education'] = 'High School'
    data.loc[data['education_num'].isin([11, 12]), 'education'] = 'Associates'
    data.loc[data['education_num'].isin([13]), 'education'] = 'Bachelors'
    data.loc[data['education_num'].isin([14]), 'education'] = 'Masters'
    data.loc[data['education_num'].isin([15, 16]), 'education'] = 'PhD/Professional'

    data = data.drop('final_weight', axis=1)
    data = data.drop_duplicates()
    data = data.drop('education_num', axis = 1)
    data = data.dropna(how='any', axis=0)
    data.capital_gain = data.capital_gain.astype(int)
   
    to_replace = ['workclass', 'education', 'marital_status', 'occupation','relationship', 'race', 'native_country']
    data = pd.get_dummies(data, columns=to_replace, drop_first = False)
   
    data['sex'] = data['sex'].astype('category').cat.codes
    data['income_class'] = data['income_class'].astype('category').cat.codes

    data['age'] = (data['age'] - data['age'].mean()) / data['age'].std()
    data['hours_per_week'] = (data['hours_per_week'] - data['hours_per_week'].mean()) / data['hours_per_week'].std()
    data['capital_gain'] = (data['capital_gain'] - data['capital_gain'].mean()) / data['capital_gain'].std()
    data['capital_loss'] = (data['capital_loss'] - data['capital_loss'].mean()) / data['capital_loss'].std()

    data = data.drop('native_country_Holand-Netherlands', axis=1)
    
    return data

def get_law():
    data = pd.read_csv('data/lawschs1_1.csv')

    data = data[data.MissingRace != 1]
    data = data.drop('Race', axis=1)
    data = data.drop('MissingRace', axis=1)
    data = data.drop('college', axis=1)
    data = data.drop('Year', axis=1)
    data = data.dropna(how='any', axis=0)

    #to_replace = ['LSAT', 'GPA']
    #data = pd.get_dummies(data, columns=to_replace, drop_first = False)
    
    return data