import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

from utilities import visualization_classifier

#definerer input data
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8],
              [5.1, 4.5], [6, 5], [5.6, 5],
                [3.3, 0.4], [3.9, 0.9], [2.8, 1],
              [0.5, 3.4], [1, 4], [0.6, 4.9]])
#Label
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

#Lag classifier
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)


#tren classifier
classifier.fit(X,y)

#visualize the data in the classifier
visualization_classifier(classifier, X, y)
