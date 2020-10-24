import numpy as np
from numpy import genfromtxt
#Importing the data as an array from txt file (Delimiter is used to separate the values into elements in an array)
data = genfromtxt('bank_note_data.txt',delimiter=',')
#The fifth column is to tell whether the bank notes are forged or not (1=authentic,0=fake)
labels = data[:,4]
print(np.count_nonzero(labels==1))
print(np.count_nonzero(labels==0))
print(labels.size)
#Grabbing the features which is the first to the fourth column
features = data[:,:4]
#Now we assign the features and labels as X and y
X = features
y = labels
#Split the data into training and test set (Automatically randomized)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)
#When working with neural networks, its good to standardize or scale your data
from sklearn.preprocessing import MinMaxScaler
scaler_object = MinMaxScaler()
#Fit the scaler object into our training data
scaler_object.fit(X_train)
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)
#Now its time to build a simple network using keras
from keras.models import Sequential
from keras.layers import Dense #Densely connected layers
model = Sequential() #Creates the model
model.add(Dense(4,input_dim=4,activation='relu')) #Add the first layer, INPUT LAYER (Activation is rectified linear unit)
model.add(Dense(5,activation='relu')) #Add the second layer, HIDDEN LAYER
model.add(Dense(1,activation='sigmoid')) #Add the last layer, OUTPUT LAYER
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#Now we train or fit the data
model.fit(scaled_X_train,y_train,epochs=50,verbose=2)
#Predict on the test data (Essentially outputs 1 or 0 according to the features)
from sklearn.metrics import confusion_matrix,classification_report
predictions = model.predict_classes(scaled_X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
model.save('C:\PythonPrograms\Deep Learning Models\mysupermodel.h5')
from keras.models import load_model
new_model = load_model('C:\PythonPrograms\Deep Learning Models\mysupermodel.h5')