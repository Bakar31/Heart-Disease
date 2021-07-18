from preprocessing import *
from sklearn.linear_model import LogisticRegression

# model-1
clf = LogisticRegression(max_iter = 1000, random_state = 31)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test)) # Score -> 1.00

from sklearn.metrics import classification_report
y_preds = clf.predict(x_test)
print(classification_report(y_test, y_preds))