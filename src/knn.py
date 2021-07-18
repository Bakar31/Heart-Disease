from preprocessing import *
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_train)
print(knn_clf.score(x_test, y_test))

from sklearn.metrics import classification_report
y_preds = knn_clf.predict(x_test)
print(classification_report(y_test, y_preds))