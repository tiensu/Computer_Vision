# USEAGE:
# python extract_faces_dataset.py --train_data 5-celebrity-faces-dataset/train/ --val_data 5-celebrity-faces-dataset/val/ --save_data 5-celebrity-faces-dataset.npz

# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import argparse

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config_tf)

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-td', '--train_data', required=True)
	ap.add_argument('-vd', '--val_data', required=True)
	ap.add_argument('-sd', '--save_data', required=True)
	args = vars(ap.parse_args())

	# load train dataset
	trainX, trainy = load_dataset(args['train_data'])
	print(trainX.shape, trainy.shape)
	# load test dataset
	testX, testy = load_dataset(args['val_data'])
	# save arrays to one file in compressed format
	savez_compressed(args['save_data'], trainX, trainy, testX, testy)

if __name__ == '__main__':
	main()