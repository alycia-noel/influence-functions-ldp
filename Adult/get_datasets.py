import numpy as np
import pandas as pd

def get_adult():
    features = ['age', 'workclass', 'final_weight', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_class']

    data = pd.read_csv('adult.data', comment='|', skipinitialspace=True, names=features,
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