from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 
from sklearn.neighbors import DistanceMetric
import numpy as np 
import matplotlib.pyplot as plt 
  
print('1 for manhattan')
print('2 for euclidean')
choice = input("Enter the number for the distance metric you want ")
metricChoice = 'manhattan'

if(choice=='1'):
	metricChoice = 'manhattan'
if(choice=='2'):
	metricChoice = 'euclidean'

irisData = load_iris() 
  
# Create feature and target arrays 
X = irisData.data #length, width 
y = irisData.target # species
#print(irisData.target_names)
  
# Split into training and test set 
X_train, X_test, y_train, y_test = train_test_split( 
             X, y, test_size = 0.2, random_state=20) 
  
neighbors = np.arange(1, 16) 
train_accuracy = np.empty(len(neighbors)) 
test_accuracy = np.empty(len(neighbors)) 
  
# Loop over K values 
for i, k in enumerate(neighbors): 
    knn = KNeighborsClassifier(n_neighbors=k, metric=metricChoice) 
    knn.fit(X_train, y_train) 
    # Compute traning and test data accuracy 
    train_accuracy[i] = knn.score(X_train, y_train) 
    test_accuracy[i] = knn.score(X_test, y_test) 
    print(test_accuracy[i])
  
# Generate plot 
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy') 

plt.legend() 
plt.xlabel('n_neighbors') 
plt.ylabel('Accuracy') 
plt.show() 