from preprocessing import *
#from prep import *
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter = 1000, random_state = 31)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
