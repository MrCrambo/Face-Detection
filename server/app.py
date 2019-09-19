import flask
import werkzeug
import time
import dlib
from imageio import imread
from scipy.spatial import distance, KDTree
from scipy.ndimage import rotate
import json

app = flask.Flask(__name__)

sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

actors_array = []
descriptors = []

return_value = "Service can not detect face!"

def load_from_json():
    with open('data2.json', 'r') as json_file:
        data = json.load(json_file)
        for p in data['actors']:
            actors_array.append(p)

    for el in actors_array:
        descriptors.append(el[2])

def find_descriptor(image):
    dets = detector(image, 1)
    print(dets)
    
    for k, d in enumerate(dets):
        shape = sp(image, d)
    
    return facerec.compute_face_descriptor(image, shape)

def find_face(descriptor):
    tree = KDTree(descriptors)
    answers = tree.query(descriptor, k=2)

    return actors_array[answers[1][0]][0]
    

@app.route('/predict', methods = ['GET', 'POST'])
def handle_request():
    files_ids = list(flask.request.files)

    imagefile = flask.request.files[files_ids[0]]
    # filename = werkzeug.utils.secure_filename(imagefile.filename)
    # print("Image Filename : " + imagefile.filename)
    # imagefile.save(filename)

    try:
        img = imread(imagefile)
        #img = rotate(img, 270)
        descr = find_descriptor(img)
    except Exception as e:
        return return_value

    return find_face(descr)

if __name__ == '__main__':
    load_from_json()

    print(actors_array[6715][0])

    app.run(host="0.0.0.0", debug=True)