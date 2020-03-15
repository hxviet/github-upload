from sklearn import tree
features = [[300, 2], [450, 2], [200, 8], [150, 9]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
horsepower = input("Vehicle's horsepower: ")
seats = input("Vehicle's number of seats: ")
vehicleType = clf.predict([[horsepower, seats]])
if (vehicleType == 0):
    print('sports-car')
if (vehicleType == 1):
    print('minivan')