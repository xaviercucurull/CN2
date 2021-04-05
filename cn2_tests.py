from CN2 import CN2
from datasets import datasets
from sklearn.metrics import classification_report

x, y = datasets.load_tennis()
#x, y = datasets.load_lenses()

cn2 = CN2(verbose=0)

cn2.fit(x, y.to_list())
cn2.print_rules()
y_pred = cn2.predict(x)

print('\n')
print(classification_report(y, y_pred))
