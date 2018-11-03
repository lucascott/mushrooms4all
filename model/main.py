from collections import defaultdict
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("model/mushrooms.csv")
print(df.head())

y = df["class"].to_frame()
X = df.drop("class", axis=1)

# Encoding the variable

attrDict = defaultdict(LabelEncoder)
yEncoder = LabelEncoder()

X = X.apply(lambda x: attrDict[x.name].fit_transform(x))
y = y.apply(lambda x: yEncoder.fit_transform(x))

'''
# Inverse the encoded
fit.apply(lambda x: attrDict[x.name].inverse_transform(x))

# Using the dictionary to label future data
df.apply(lambda x: attrDict[x.name].transform(x))
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

mlp = MLPClassifier()
mlp.fit(X_train, y_train)
print("Accuracy: " + str(mlp.score(X_test, y_test)))
y_prob = mlp.predict_proba(X_test)[:, 1]
y_pred = np.where(y_prob > 0.5, 1, 0)
print(y_pred)

# save the model to disk
filenameEnc = 'model/encoder.sav'
pickle.dump(attrDict, open(filenameEnc, 'wb'))
filenameYEnc = 'model/Yencoder.sav'
pickle.dump(yEncoder, open(filenameYEnc, 'wb'))

filename = 'model/mpl.sav'
pickle.dump(mlp, open(filename, 'wb'))
