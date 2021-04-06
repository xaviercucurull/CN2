from CN2 import CN2
from datasets import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import time

#x, y = datasets.load_tennis()
#x, y = datasets.load_lenses()



# x, y = datasets.load_csv('datasets/play_tennis.csv', print_summary=True)


def test_small():
    # Load Heart Disease database from CSV
    # https://archive.ics.uci.edu/ml/datasets/Heart+Disease
    print('##############################################')
    print('###########  Test Small Dataset ##############')
    print('###########     Heart Disease   ##############')
    print('##############################################\n')
    x, y = datasets.load_csv('datasets/heart.csv')
    test_dataset(x, y)

def test_medium():
    # Load Mammographic Mass database
    print('##############################################')
    print('########### Test Medium Dataset ##############')
    print('###########  Mammographic Mass  ##############')
    print('##############################################\n')
    x, y = datasets.load_mammographic_mass()
    test_dataset(x, y)

def test_dataset(x, y):
    """ Given a dataset (data features and classes), split it into train and test 
    and evaluate the CN2 induction algorithm.
    
    TODO 
    Print relevant information such as...

    Args:
        x (DataFrame): training data features
        y (Series): training data classification
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
    cn2.fit(X_train, y_train.to_list())
    time_fit = time.time() - time0
    print('CN2 trained in {:.1f}s'.format(time_fit))
    print('{} rules obtained:\n'.format(len(cn2.rules_list)))
    cn2.print_rules()
    
    # TODO build a dataFrame of rules + coverage/precision -> to_latex
    
    # predict test data
    time0 = time.time()
    y_pred = cn2.predict(X_test)
    time_predict = time.time() - time0
    print('\nCN2 prediction made in {:.1f}s\n'.format(time_predict))

    print('Classification report:')
    print(classification_report(y_test, y_pred.astype(y_test.dtype)))
    
test_medium()
test_small()