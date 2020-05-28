import numpy as np
import onnxruntime
from pydantic import BaseModel
import uvicorn
import pickle
from fastapi import FastAPI
from sklearn.ensemble import RandomForestClassifier

#Create an object of class FastAPI
global app
app=FastAPI()

#Load feature_names and target name
feature_names=pickle.load(open('app/features.pickle','rb'))
		
target_name=pickle.load(open('app/target.pickle','rb'))
		
#Now in onnx model we have to create an InferenceSession and pass the onnx file path in it
session=onnxruntime.InferenceSession('app/rf_m.onnx')

#Now we get the input and out names we used to save the model initially.
first_input_name=session.get_inputs()[0].name
		
first_output_name=session.get_outputs()[0].name

class Data(BaseModel):
	'''In fast-api this class is created just for documentation purposes'''
	sepal_length: float
	sepal_width: float
	petal_length: float
	petal_width: float


@app.post("/predict")
def predict(data:Data):
	try:
		#Conver the json into dict
		data_dict=data.dict()
		#Use the feature name to make the dict into a list
		
		to_predict=[data_dict[feature] for feature in feature_names]
		#Reshape the data into 1 row and n columns where n is the number of features in the dataset.In iris it is 4

		to_predict=np.array(to_predict).reshape(1,-1)
		#Now run the session and since while creating we made sure the input is in flat we explicitly convert the data into float
		pred_onx = session.run([first_output_name], {first_input_name:to_predict.astype(np.float32)})[0]
		
		#Now we map these labels back to the class names.In case of iris it will setosa,virginica and versicolor.
		encode2label=target_name[int(pred_onx[0])]

		return {"prediction":str(encode2label)}
	except:
		return {"prediction": "error"}
