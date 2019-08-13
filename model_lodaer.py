import keras


from keras.models import load_model
import boto3
import botocore
import numpy 

def download_file(bucket_name,file_name,output_filename):
	s3 = boto3.resource('s3')
	try:
		s3.Bucket(bucket_name).download_file(file_name, output_filename)
	except botocore.exceptions.ClientError as e:
    	if e.response['Error']['Code'] == "404":
    		print("The object does not exist.")
		else:
        	raise

def load_model(bucket_name,model_path):
	download_file(bucket_name,model_path,"model.h5")
	model = load_model('model.h5')
	return model
	
def evaluvate_model(dataset_bucket,dataset_path,model_bucket,model_path):
	download_file(bucket,dataset_path+".X","X.npy")
	download_file(bucket,dataset_path+".Y","Y.npy")
	model = load_model(model_bucket,model_path)
	X_test = numpy.load("X.npy")
	Y_test = numpy.load("Y.npy")
	score = model.evaluvate(X_test,Y_test)
	return score;

def create_bucket(bucket_name):
    s3 = boto3.client('s3')
    s3.create_bucket(Bucket=bucket_name)
    
def upload_file(entity_bucket,entity_name,file_path):
    s3 = boto3.client('s3')
    s3.upload_file(entity_name,entity_bucket, file_path)
    
    
    