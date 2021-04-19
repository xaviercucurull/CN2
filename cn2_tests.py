"""
Supervised and Experiential Learning (SEL)
Master in Artificial Intelligence (UPC)
PW1 - Implementation of the CN2 Induction Algorithm

Author: Xavier Cucurull Salamero <xavier.cucurull@estudiantat.upc.edu>
Course: 2020/2021
"""
import sys
import os

from datasets import datasets
from CN2 import CN2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import time


def test_lenses():
    # Load Lenses dataset
    print('##############################################')
    print('#############  Test Toy Dataset ##############')
    print('#############       Lenses      ##############')
    print('##############################################\n')
    x, y = datasets.load_lenses()
    cn2 = CN2()
    print('Training CN2 induction algorithm...')
    time0 = time.time()
    cn2.fit(x, y)
    time_fit = time.time() - time0
    print('CN2 trained in {:.1f}s'.format(time_fit))
    y_pred = cn2.predict(x)
    cn2.print_rules()
    print('Classification report:')
    print(classification_report(y, y_pred.astype(y.dtype)))
    return cn2

def test_small(n_bins=4,  print_rules=True):
    # Load Heart Disease database from CSV
    # https://www.kaggle.com/ronitf/heart-disease-uci
    print('##############################################')
    print('###########  Test Small Dataset ##############')
    print('###########     Heart Disease   ##############')
    print('##############################################\n')
    x, y = datasets.load_csv(os.path.join('datasets','DATA', 'heart.csv'))
    cn2 = test_dataset(x, y, n_bins=n_bins, print_rules=print_rules)
    return cn2

def test_medium(n_bins=4,  print_rules=True):
    # Load Mammographic Mass dataset
    print('##############################################')
    print('########### Test Medium Dataset ##############')
    print('###########  Mammographic Mass  ##############')
    print('##############################################\n')
    x, y = datasets.load_mammographic_mass()
    cn2 = test_dataset(x, y, n_bins=n_bins, print_rules=print_rules)
    return cn2

def test_large(n_bins=4,  print_rules=True):
    # Load Rice dataset
    print('##############################################')
    print('############# Test Large Dataset #############')
    print('#################   Rice  ###################')
    print('##############################################\n')
    x, y = datasets.load_rice()
    cn2 = test_dataset(x, y, n_bins=n_bins, print_rules=print_rules)
    return cn2
    
def test_dataset(x, y, n_bins=4, print_rules=True):
    """ Given a dataset (data features and classes), split it into train and test 
    and evaluate the CN2 induction algorithm.

    Args:
        x (DataFrame): training data features
        y (Series): training data classification
        n_bins (int, optional): number of bins used for discretization of continuous attributes. Defaults to 4.
    """
    cn2 = CN2()

    # split data into 75% train and 25% test
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    print('Data attributes: {}'.format(len(X_train.keys())))
    print('Training size: {}'.format(len(y_train)))
    print('Test size: {}\n'.format(len(y_test)))

    # train CN2 and display obtained rules
    print('Training CN2 induction algorithm...')
    time0 = time.time()
    cn2.fit(X_train, y_train.to_list(), n_bins)
    time_fit = time.time() - time0
    print('CN2 trained in {:.1f}s'.format(time_fit))
    print('{} rules obtained:\n'.format(len(cn2.rules_list)))
    if print_rules:
        cn2.print_rules()
        
    # predict test data
    time0 = time.time()
    y_pred = cn2.predict(X_test)
    time_predict = time.time() - time0
    print('\nCN2 prediction made in {:.1f}s\n'.format(time_predict))

    print('Classification report:')
    print(classification_report(y_test, y_pred.astype(y_test.dtype)))
    
    return cn2

if __name__ == "__main__":
    # Execute tests   
    cn_lenses = test_lenses()
    cn_s = test_small()
    cn_m =  test_medium()
    cn_l = test_large()

    out_path = os.path.join('Out')

    # Save generated tables        
    with open(os.path.join(out_path, 'rules_lenses.csv'), 'w') as tf:
        df = cn_lenses.generate_rules_table()
        tf.write(df.to_csv())

    with open(os.path.join(out_path, 'rules_small.csv'), 'w') as tf:
        df = cn_s.generate_rules_table()
        tf.write(df.to_csv())
        
    with open(os.path.join(out_path, 'rules_medium.csv'), 'w') as tf:
        df = cn_m.generate_rules_table()
        tf.write(df.to_csv())
        
    with open(os.path.join(out_path, 'rules_large.csv'), 'w') as tf:
        df = cn_l.generate_rules_table()
        tf.write(df.to_csv())
        
    # Save rules to text file
    cn_s.save_rules(os.path.join(out_path, 'rules_small.txt'))
    cn_m.save_rules(os.path.join(out_path, 'rules_medium.txt'))
    cn_l.save_rules(os.path.join(out_path, 'rules_large.txt'))
    cn_lenses.save_rules(os.path.join(out_path, 'rules_lenses.txt'))
