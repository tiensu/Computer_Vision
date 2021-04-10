# python extract_faces.py --dp '5-celebrity-faces-dataset/train/ben_afflek/'
# demonstrate face detection on 5 Celebrity Faces Dataset
from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import argparse

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

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

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-dp', '--data_path', required=True)
	args = vars(ap.parse_args())

	i = 1
	# enumerate files
	for filename in listdir(args['data_path']):
		# path
		path = args['data_path'] + filename
		# get face
		face = extract_face(path)
		print(i, face.shape)
		# plot
		pyplot.subplot(2, 7, i)
		pyplot.axis('off')
		pyplot.imshow(face)
		i += 1
	pyplot.show()

if __name__ == '__main__':
	main()
