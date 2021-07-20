from preprocessing import *
from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(random_state = 31)
rand_clf.fit(x_train, y_train)
print(rand_clf.score(x_test, y_test)) # Score -> 1.00

from sklearn.metrics import classification_report
y_preds = rand_clf.predict(x_test)
print(classification_report(y_test, y_preds))