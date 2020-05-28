from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pandas as pd
import pickle

def load_data():
	"""This function will save the feature names
	and target name
	
	Returns:X_train and Y_train
	"""
	data=load_iris()
	X=data['data']
	Y=data['target']
	target_name=data['target_names']

	feature_names=data['feature_names']
	'''By default the feature names are sepal length (cm) and we will simplify it to sepal_length 
	in the name so we will replace it with _'''
	feature_names=[' '.join(name.split()[:2]).replace(' ','_') for name in feature_names]
	#Save the target_name and feature names
	pickle.dump(target_name,open('app/target.pickle','wb'))
	pickle.dump(feature_names,open('app/features.pickle','wb'))
	return X,Y

def save_onnx():

	X,Y=load_data()

	'''Here A basic model is used but more hyperparameter tuning can be 
	done'''

	rf=RandomForestClassifier()
	rf.fit(X,Y)

	#Now we have to save the model in onnx format
	initial_type = [('float_input', FloatTensorType([None,X.shape[1]]))]
	onx = convert_sklearn(rf, initial_types=initial_type)
	with open('app/rf_m.onnx', "wb") as f:
		f.write(onx.SerializeToString())

if __name__ == '__main__':
    save_onnx()