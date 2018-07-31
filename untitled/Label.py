import numpy as np
from sklearn import preprocessing

#sample labels
inputLabels = ['red', 'black','red','green','black','yellow','white','blue']

#Create label encoder
encoder = preprocessing.LabelEncoder()
encoder.fit(inputLabels)
print("\nLabel mapping:")
for i, label in enumerate(encoder.classes_):
    print(label, '-->', i)

test_labels = ['blue', 'red', 'yellow', 'black']
encodedValues = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("encoded values =", list(encodedValues))

#decoding numbers
encodedValues = [3, 0, 4, 1]
decodedlist = encoder.inverse_transform(list(encodedValues))
print("\nEncoded values =", encodedValues)
print("Decoded labels =", list(decodedlist))
