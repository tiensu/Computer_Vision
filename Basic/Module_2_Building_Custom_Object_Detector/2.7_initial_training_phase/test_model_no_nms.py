# USAGE
# python test_model_no_nms.py --conf conf/cars.json \
#	--image datasets/caltech101/101_ObjectCategories/car_side/image_0035.jpg

# import the necessary packages
from object_detection import ObjectDetector
from descriptors import HOG
from utils import Conf
import imutils
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
ap.add_argument("-i", "--image", required=True, help="path to the image to be classified")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# load the classifier, then initialize the Histogram of Oriented Gradients descriptor
# and the object detector
model = pickle.loads(open(conf["classifier_path"], "rb").read())
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
	cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"], block_norm="L1")
od = ObjectDetector(model, hog)

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=min(260, image.shape[1]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect objects in the image
(boxes, probs) = od.detect(gray, conf["window_dim"], winStep=conf["window_step"],
	pyramidScale=conf["pyramid_scale"], minProb=conf["min_probability"])

# loop over the bounding boxes and draw them
for (startX, startY, endX, endY) in boxes:
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

# show the output images
cv2.imshow("Image", image)
cv2.waitKey(0)