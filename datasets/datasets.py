import os
import pandas as pd
import numpy as np

def load_tennis(print_summary=False):
    """ Load play tennis dataset

    Returns:
        [type]: [description]
    """
    # Load CSV dataset as pandas table
    df = pd.read_csv('datasets/play_tennis.csv', delimiter=',')
    df.dataframeName = 'Play Tennis'
    
    # Remove day column (not a feature)
    df.drop('day', 1, inplace=True)
    
    # Dataset summary
    if print_summary:
        for c in df.columns:
            if c!='play':
                print(df.groupby([c, 'play']).size())
                print('------------------')
        df['play'].value_counts()
    
    # Separate data and class
    df_class = df['play']
    df.drop('play', 1, inplace=True)
    
    return df, df_class

def load_lenses(print_summary=False):
    """ Load lenses dataset
    
    Attribute Information:
    -- 3 Classes
    1 : the patient should be fitted with hard contact lenses,
    2 : the patient should be fitted with soft contact lenses,
    3 : the patient should not be fitted with contact lenses.

    1. age of the patient: (1) young, (2) pre-presbyopic, (3) presbyopic
    2. spectacle prescription: (1) myope, (2) hypermetrope
    3. astigmatic: (1) no, (2) yes
    4. tear production rate: (1) reduced, (2) normal
    
    Returns:
        [type]: [description]
    """
    # Load DATA dataset as pandas table
    lenses_header = ['age', 'deficiency', 'astigmatic', 'tear production', 'recommendation']
    df = pd.read_table('datasets/lenses.data', delimiter='\s+', index_col=0, names=lenses_header)
    df.dataframeName = 'Lenses'
    
    # Replace attributes
    replace_dicts = {'age': {1: 'young', 2: 'pre-presbyopic', 3:'presbyopic'},
                     'deficiency': {1: 'myope', 2: 'hypermetrope'},
                     'astigmatic': {1: 'no', 2: 'yes'},
                     'tear production': {1: 'reduced', 2: 'normal'},
                     'recommendation': {1: 'hard', 2: 'soft', 3: 'none'}}
    
    for k in replace_dicts.keys():
        df[k].replace(replace_dicts[k], inplace=True)
    
    # Dataset summary
    if print_summary:
        for c in df.columns:
            if c!='recommendation':
                print(df.groupby([c, 'recommendation']).size())
                print('------------------')
        df['recommendation'].value_counts()
        
    # Separate data and class
    df_class = df['recommendation']
    df.drop('recommendation', 1, inplace=True)
    
    return df, df_class