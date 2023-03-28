import math
import torch
import random

import pandas as pd

from collections import OrderedDict
from torch.utils.data import Dataset

E = math.e

class TabularData(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y)
        n, m = x.shape
        self.n = n
        self.m = m
        self.x = x
        self.y = y
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def read_dataset(path, data_types):
    data = pd.read_csv(
        path,
        names=data_types,
        index_col=None,
        dtype=data_types,
        comment='|',
        skipinitialspace=True,
        na_values={
            'capital_gain': 99999,
            'workclass': '?',
            'native_country': '?',
            'occupation': '?'
        },
    )

    return data

def clean_and_encode_dataset(data):
    features = ['age', 'workclass', 'final_weight', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'captial_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_class']
    data = data.replace({'<=50K.' : '<=50K', '>50K.' : '>50K'})
    
    # drop entries where workclass and occupation are unknown, and where workclass is without pay
    data = (data[(data['workclass'] != '?') & (data['occupation'] != '?') & (data['workclass'] != 'Without-pay')].reset_index(drop=True))

    data.loc[data['occupation'] == 'Armed-Forces', 'occupation'] = 'Protective-serv'

#     data.loc[data['workclass'].isin(['State-gov', 'Federal-gov', 'Local-gov']), 'workclass'] = 'Government'
#     data.loc[data['workclass'].isin(['Self-emp-not-inc', 'Self-emp-inc']), 'workclass'] = 'Self-Employed'
#     data.loc[data['workclass'].isin(['Private']), 'workclass'] = 'Privately-Employed'

#     data.loc[data['marital_status'].isin(['Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent']), 'marital_status'] ='Married'
#     data.loc[data['marital_status'].isin(['Divorced', 'Never-married', 'Separated', 'Widowed']), 'marital_status'] = 'Not-married'

    data.loc[data['education_num'] <= 8, 'education'] = 'Less than High School'
    data.loc[data['education_num'].isin([9, 10]), 'education'] = 'High School'
    data.loc[data['education_num'].isin([11, 12]), 'education'] = 'Associates'
    data.loc[data['education_num'].isin([13]), 'education'] = 'Bachelors'
    data.loc[data['education_num'].isin([14]), 'education'] = 'Masters'
    data.loc[data['education_num'].isin([15, 16]), 'education'] = 'PhD/Professional'
  
    data = data.drop('final_weight', axis=1)
    
    frames = []
    
#     data = data.drop('education', axis=1)
    
#     data = data.drop('native_country', axis=1)
#     data = data.drop('relationship', axis=1)
    data = data.drop_duplicates()
    data = data.drop('education_num', axis = 1)
    data = data.dropna(how='any', axis=0)
    data.capital_gain = data.capital_gain.astype(int)

    data['capital_diff'] = abs((data['capital_gain'] - data['capital_loss']))
    data = data.drop('capital_gain', axis=1)
    data = data.drop('capital_loss', axis=1)
    
    #data['workclass'] = data['workclass'].astype('category').cat.codes 
    ohe_workclass = pd.get_dummies(data.workclass, prefix='')
    data = data.drop('workclass', axis=1)
    frames.append(ohe_workclass)
    
    #data['education'] = data['education'].astype('category').cat.codes 
    ohe_education = pd.get_dummies(data.education, prefix='')
    data = data.drop('education', axis=1)
    frames.append(ohe_education)
    
    #data['marital_status'] = data['marital_status'].astype('category').cat.codes
    ohe_ms = pd.get_dummies(data.marital_status, prefix='')
    data = data.drop('marital_status', axis=1)
    frames.append(ohe_ms)
    
    #data['occupation'] = data['occupation'].astype('category').cat.codes
    ohe_occ = pd.get_dummies(data.occupation, prefix='')
    data = data.drop('occupation', axis=1)
    frames.append(ohe_occ)
    
    #data['relationship'] = data['relationship'].astype('category').cat.codes
    ohe_relationship = pd.get_dummies(data.relationship, prefix='')
    data = data.drop('relationship', axis=1)
    frames.append(ohe_relationship)
    
    #data['race'] = data['race'].astype('category').cat.codes 
    ohe_race = pd.get_dummies(data.race, prefix='')
    data = data.drop('race', axis=1)
    frames.append(ohe_race)
    
    data['sex'] = data['sex'].astype('category').cat.codes  
    #ohe_sex = pd.get_dummies(data.sex, prefix='')
    #data = data.drop('sex', axis=1)
    #frames.append(ohe_sex)
    
    #data['native_country'] = data['native_country'].astype('category').cat.codes
    ohe_nc = pd.get_dummies(data.native_country, prefix='')
    data = data.drop('native_country', axis=1)
    frames.append(ohe_nc)
    
    data['income_class'] = data['income_class'].astype('category').cat.codes

    # Normalize continuous data
    #data['age'] = (data['age'] - data['age'].mean()) / data['age'].std()
    #data['hours_per_week'] = (data['hours_per_week'] - data['hours_per_week'].mean()) / data['hours_per_week'].std()
    #data['education_num'] = (data['education_num'] - data['education_num'].mean()) / data['education_num'].std()
    #data['capital_gain'] = (data['capital_gain'] - data['capital_gain'].mean()) / data['capital_gain'].std()
    #data['capital_loss'] = (data['capital_loss'] - data['capital_loss'].mean()) / data['capital_loss'].std()
    
    age_mean = data['age'].mean()
    data.loc[data['age'] < age_mean, 'age'] = 0
    data.loc[data['age'] >= age_mean, 'age'] = 1
    
    hpw_mean = data['hours_per_week'].mean()
    data.loc[data['hours_per_week'] < hpw_mean, 'hours_per_week'] = 0
    data.loc[data['hours_per_week'] >= hpw_mean, 'hours_per_week'] = 1
    
#     edu_mean = data['education_num'].mean()
#     data.loc[data['education_num'] < edu_mean, 'education_num'] = 0
#     data.loc[data['education_num'] >= edu_mean, 'education_num'] = 1
    
#     cg_mean = data['capital_gain'].mean()
#     data.loc[data['capital_gain'] < cg_mean, 'capital_gain'] = 0
#     data.loc[data['capital_gain'] >= cg_mean, 'capital_gain'] = 1

#     cl_mean = data['capital_loss'].mean()
#     data.loc[data['capital_loss'] < cl_mean, 'capital_loss'] = 0
#     data.loc[data['capital_loss'] >= cl_mean, 'capital_loss'] = 1
    
    cl_diff = data['capital_diff'].mean()
    data.loc[data['capital_diff'] < cl_diff, 'capital_diff'] = 0
    data.loc[data['capital_diff'] >= cl_diff, 'capital_diff'] = 1
    
    frames.append(data)
    full_data = pd.concat(frames, axis=1)

    # drop holland-netherland since not in test
    if len(full_data.columns) == 89:
        full_data = full_data.drop('_Holand-Netherlands', axis=1)
      
    return full_data
    
def get_dataset():
    train_data_file = 'data/adult.data'
    test_data_file = 'data/adult.test'

    data_types = OrderedDict([
        ("age", "int"),
        ("workclass", "str"),
        ("final_weight", "int"),
        ("education", "str"),
        ("education_num", "int"),
        ("marital_status", "str"),
        ("occupation", "str"),
        ("relationship", "str"),
        ("race", "str"),
        ("sex", "str"),
        ("capital_gain", "float"),
        ("capital_loss", "int"),
        ("hours_per_week", "int"),
        ("native_country", "str"),
        ("income_class", "str"),
    ])

    train_data = clean_and_encode_dataset(read_dataset(train_data_file, data_types))
    test_data = clean_and_encode_dataset(read_dataset(test_data_file, data_types))

    datasets = [train_data, test_data]

    cols = train_data.columns
    
    features, labels = cols[:-1], cols[-1]
    
    return datasets, features, labels

def get_data_adult():
    datasets, features, labels = get_dataset()
    train_data, test_data = datasets

#     train_dataset = TabularData(train_data[features].values, train_data[labels].values)
#     test_dataset = TabularData(test_data[features].values, test_data[labels].values)

    return len(features), [train_data[features].values, train_data[labels].values], [test_data[features].values, test_data[labels].values]