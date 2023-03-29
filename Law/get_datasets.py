import numpy as np
import pandas as pd
import fairlearn 

def get_law():
    data = pd.read_csv('lawschs1_1.csv')
    
    data = data[data.MissingRace != 1]
    data = data.drop('Race', axis=1)
    data = data.drop('MissingRace', axis=1)
    data = data.drop('college', axis=1)
    data = data.drop('Year', axis=1)
    data = data.dropna(how='any', axis=0)

    data['LSAT'] = data['LSAT'].apply(lambda x: round(x/10))
    data['GPA'] = data['GPA'].apply(lambda x: round(x, 1))
    
    to_replace = ['LSAT', 'GPA']
    data = pd.get_dummies(data, columns=to_replace, drop_first = False)
    
    
    return data