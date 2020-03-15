from scipy.spatial import distance
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    def closest(self, row):
        best_dist = distance.euclidean(row, self.X_train[0])
        best_index = 0
        for index in range(1, len(self.X_train)):
            dist = distance.euclidean(row, self.X_train[index])
            if dist < best_dist:
                best_dist = dist
                best_index = index
        return self.y_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

question = "Decision tree ('0'), KNeighbors ('1'), or ScappyKNN ('2')? "
clf_type = input(question)
while clf_type != '0' and clf_type != '1' and clf_type != '2':
    clf_type = input(question)
if clf_type == '0':
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
if clf_type == '1':
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
if clf_type == '2':
    clf = ScrappyKNN()
    
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('\n', predictions)

from sklearn.metrics import accuracy_score
print('\n Accuracy:', accuracy_score(y_test, predictions))