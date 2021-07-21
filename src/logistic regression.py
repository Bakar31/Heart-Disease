from preprocessing import *
from sklearn.linear_model import LogisticRegression

# model-1
clf = LogisticRegression(max_iter = 1000, random_state = 31)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

from sklearn.metrics import classification_report
y_preds = clf.predict(x_test)
print(classification_report(y_test, y_preds))

devx = dev.drop('target', axis = 1)
print(devx.head())
devy = dev['target']

# to test overfitting
dev_preds = clf.predict(devx)
print(clf.score(devx, devy))
print(classification_report(devy, dev_preds))