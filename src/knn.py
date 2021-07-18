from preprocessing import *
from sklearn.neighbors import KNeighborsClassifier

# Model-2
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_train)
print(knn_clf.score(x_test, y_test)) # Score -> 0.945

from sklearn.metrics import classification_report
y_preds = knn_clf.predict(x_test)
print(classification_report(y_test, y_preds))