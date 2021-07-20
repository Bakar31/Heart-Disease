from preprocessing import *

from sklearn import svm
svc_clf = svm.SVC(random_state = 7)
svc_clf.fit(x_train, y_train)
print(svc_clf.score(x_test, y_test))