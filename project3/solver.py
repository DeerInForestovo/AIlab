from sklearn import tree
import matplotlib
import csv

# versions:
#   python: 3.9
#   sklearn: 0.0.post5
#   matplotlib: 3.7.1

if __name__ == "__main__":
    # read from files
    train_data_filename = "./data/traindata.csv"
    train_label_filename = "./data/trainlabel.txt"
    test_data_filename = "./data/testdata.csv"
    test_label_filename = "./data/testlabel.txt"
    title = None
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    with open(file=train_data_filename) as tdf:
        for row in csv.reader(tdf):
            row = [int(item) if item.isdecimal() else item for item in row]
            if title is None:
                title = row
            else:
                train_data.append(row)

    with open(file=train_label_filename) as tlf:
        for row in tlf:
            train_label.append(int(row))

    title = None

    with open(file=test_data_filename) as tdf:
        for row in csv.reader(tdf):
            row = [int(item) if item.isdecimal() else item for item in row]
            if title is None:
                title = row
            else:
                test_data.append(row)

    # train
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_label)
    score = clf.score(train_data, train_label)
    print("score = ", score)

    # predict
    test_label = clf.predict(test_data)
    print(test_label)

    pass

