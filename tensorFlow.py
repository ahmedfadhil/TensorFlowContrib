from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow.contrib.learn.python.learn as learn
from sklearn.metrics import confusion_matrix, classification_report

iris = load_iris()
iris.keys()

X = iris['data']
y = iris['target']

y.dtype()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
