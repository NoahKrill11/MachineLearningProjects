from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


def MLP(df):
	feature = ['mean radius', 'mean texture']
	x=df[feature]
	y=df.target
	trans = Pipeline(steps=[('normalize', StandardScaler()), ('model', LogisticRegression())])
	trans.fit(x,y)
	print(trans.score(x,y))
	
	
	X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y,random_state=1) 

	choice = input("How many hidden layers: ")
	choice2 = input("R for relu and L for logistic ")
	choice3 = input("Gradient decent solvers  l for lbfgs, s for sgd, a for adam ")

	solver = ''
	activation = ''


	if(choice2 =='r' or choice2=='R'):
		activation = 'relu'
	if(choice2 =='l' or choice2=='L'):
		activation = 'logistic'

	if(choice3=='l'):
		solver = 'lbfgs'
	if(choice3=='s'):
		solver = 'sgd'
	if(choice3=='a'):
		solver = 'adam'


	if(choice=='1'):
	
		mlp = MLPClassifier(hidden_layer_sizes=(10), activation=activation, solver=solver, max_iter=1600, alpha = 0.00001)
		mlp.fit(X_train,y_train)
		predict_train = mlp.predict(X_train)
		predict_test= mlp.predict(X_test)
		#print('10 links')
		print(confusion_matrix(y_train,predict_train))
		print(classification_report(y_train,predict_train))


		mlp = MLPClassifier(hidden_layer_sizes=(5), activation=activation, solver=solver, max_iter=1600, alpha = .1)
		mlp.fit(X_train,y_train)
		predict_train = mlp.predict(X_train)
		predict_test= mlp.predict(X_test)
		#print('10 links')
		print(confusion_matrix(y_train,predict_train))
		print(classification_report(y_train,predict_train))

	if(choice=='2'):
		mlp = MLPClassifier(hidden_layer_sizes=(5,10), activation=activation, solver=solver, max_iter=600, alpha = 0.0001)
		mlp.fit(X_train,y_train)
		predict_train = mlp.predict(X_train)
		predict_test= mlp.predict(X_test)
		
		print(confusion_matrix(y_train,predict_train))
		print(classification_report(y_train,predict_train))

		mlp = MLPClassifier(hidden_layer_sizes=(10,5), activation=activation, solver=solver, max_iter=600, alpha = 0.0001)
		mlp.fit(X_train,y_train)
		predict_train = mlp.predict(X_train)
		predict_test= mlp.predict(X_test)
		
		print(confusion_matrix(y_train,predict_train))
		print(classification_report(y_train,predict_train))

	if(choice=='3'):
		mlp = MLPClassifier(hidden_layer_sizes=(5,10,5), activation=activation, solver=solver, max_iter=600, alpha = 0.0001)
		mlp.fit(X_train,y_train)
		predict_train = mlp.predict(X_train)
		predict_test= mlp.predict(X_test)
		print(confusion_matrix(y_train,predict_train))
		print(classification_report(y_train,predict_train))

		mlp = MLPClassifier(hidden_layer_sizes=(10,20,5), activation=activation, solver=solver, max_iter=600, alpha = 0.0001)
		mlp.fit(X_train,y_train)
		predict_train = mlp.predict(X_train)
		predict_test= mlp.predict(X_test)
		print(confusion_matrix(y_train,predict_train))
		print(classification_report(y_train,predict_train))


cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns= np.append(cancer['feature_names'], ['target']))
MLP(df)



