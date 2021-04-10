# USEAGE:
# python random_face_identity_classification.py --face_dataset 5-celebrity-faces-dataset.npz --face_embedding 5-celebrity-faces-dataset.npz

# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-fd', '--face_dataset', required=True)
    ap.add_argument('-fe', '--face_embedding', required=True)
    args = vars(ap.parse_args())

    # load faces
    data = load(args['face_dataset'])
    testX_faces = data['arr_2']
    # load face embeddings
    data = load(args['face_embedding'])
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    # predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)
    # score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))


    # test model on a random example from the test dataset
    selection = choice([i for i in range(testX.shape[0])])
    random_face_pixels = testX_faces[selection]
    random_face_emb = testX[selection]
    random_face_class = testy[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])
    # prediction for the face
    samples = expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    print('Expected: %s' % random_face_name[0])
    # plot for fun
    pyplot.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    pyplot.title(title)
    pyplot.show()

if __name__ == '__main__':
    main()