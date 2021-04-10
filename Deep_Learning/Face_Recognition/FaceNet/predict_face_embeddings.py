# USEAGE:
# python predict_face_embeddings.py --face_dataset 5-celebrity-faces-dataset.npz --facenet_model facenet_keras.h5 --face_embedding 5-celebrity-faces-embeddings.npz

# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from tensorflow.keras.models import load_model
import argparse

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config_tf)

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def main():
	# load the face dataset
	data = load(args['face_dataset'])
	trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
	print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
	# load the facenet model
	model = load_model(args['facenet_model'])
	print('Loaded Model')
	# convert each face in the train set to an embedding
	newTrainX = list()
	for face_pixels in trainX:
		embedding = get_embedding(model, face_pixels)
		newTrainX.append(embedding)
	newTrainX = asarray(newTrainX)
	print(newTrainX.shape)
	# convert each face in the test set to an embedding
	newTestX = list()
	for face_pixels in testX:
		embedding = get_embedding(model, face_pixels)
		newTestX.append(embedding)
	newTestX = asarray(newTestX)
	print(newTestX.shape)
	# save arrays to one file in compressed format
	savez_compressed(args['face_embedding'], newTrainX, trainy, newTestX, testy)

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-fd', '--face_dataset', required=True)
	ap.add_argument('-fnm', '--facenet_model', required=True)
	ap.add_argument('-fe', '--face_embedding', required=True)
	args = vars(ap.parse_args())

	main()