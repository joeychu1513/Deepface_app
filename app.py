import os
import datetime

from flask import Flask, flash, render_template, request, session, url_for
from flask_session import Session
from deepface import DeepFace
from werkzeug.utils import secure_filename

from deepface import DeepFace


# NEED TO CHANGE THE LOCAL PATH
LOCAL_PATH = "C:/Users/user/Documents/bitbucket/joey_own_project/deepface_app"

# NEED TO DELETE THE .pkl file if photo in the DB is changed
DB_PATH = LOCAL_PATH + "/static/us_president_photos_edited"

UPLOAD_FOLDER = LOCAL_PATH + "/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configure application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        img_path = ""
        ### get image input
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template("error.html", message="No file part")
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template("error.html", message="No selected file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            ### find local path
            img_path = UPLOAD_FOLDER + "/" + filename

            # print(allowed_file(file.filename))
            # print(img_path)
            # return render_template("error.html", message="Success")
        else:
            return render_template("error.html", message="Only support JPG, PNG, JPEG")
        
        ### perform deepface find
        try:
            results, unique_results = recognize(img_path)
            ### calculate likelihood
            combined_results = likelihood(results, unique_results)

            ### not similar to presidents
            if len(combined_results) == 0:
                return render_template("error.html", message="Not similar to the Presidents")
            else:
                return render_template("result.html", combined_results=combined_results)
                
        ### no face detect or errer
        except:
            return render_template("error.html", message="No face detected or more than one face")

        ### redirect to result page

        # return render_template("result.html", )
        


    ### request method == get

    else:

        ### returen index.html
        return render_template("index.html")

def recognize(image_path):

    models = [
    "VGG-Face", 
    "Facenet", 
    "Facenet512", 
    "OpenFace", 
    "DeepFace", 
    "DeepID", 
    "ArcFace", 
    "Dlib", 
    "SFace",
    ]

    detectors = [
    'opencv', 
    'ssd', 
    'dlib', 
    'mtcnn', 
    'retinaface', 
    'mediapipe'
    ]

    metrics = ["cosine", "euclidean", "euclidean_l2"]

    ### Face Verification

    # img1_path = "testing/DM1.jpg"
    # img2_path = "testing/DM2.jpg"

    # print(img1_path)
    # print(img2_path)

    # result = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, model_name = models[1], detector_backend = detectors[1], distance_metric = metrics[0])
    # print(result)

    ### test other function

    # embedding = DeepFace.represent(img_path = img1_path)

    # print(DB_PATH)
    df = DeepFace.find(img_path = image_path, db_path = DB_PATH, model_name = models[1], detector_backend = detectors[1])

    print(df.to_dict())

    # print(len(df.to_dict())) # len of dictionary
    # print(len(df.to_dict()["identity"])) # len of found result

    ### simplify results

    results = []
    unique_results = set()


    for i in range(len(df.to_dict()["identity"])):
    ### simplify result path
        result = df.to_dict()["identity"][i].split("\\")[1].split("/")[0].replace("_", " ")
        results.append(result)
        unique_results.add(result)

    print(results)
    print(unique_results)
    return results, unique_results

def likelihood(results, unique_results):
    ### results is list, unique_results is set, need len(results) > 0

    combined_results = []

    if len(unique_results) == 0:
        return combined_results
    elif len(unique_results) == 1:
        if len(results) > 1:
            combined_results.append({"name": list(unique_results)[0], "likelihood": "extremely likely"})
        else:
            combined_results.append({"name": list(unique_results)[0], "likelihood": "very likely"})
    else:
        for item in list(unique_results):
            count_on_results = results.count(item)
            if count_on_results > 1:
                combined_results.append({"name": item, "likelihood": "likely"})
            else:
                combined_results.append({"name": item, "likelihood": "maybe"})

    return combined_results