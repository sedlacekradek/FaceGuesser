from flask import Flask, render_template, request, abort
import os
from werkzeug.utils import secure_filename
import eval


# - - - CONSTANTS - - -
UPLOAD_FOLDER = 'static/uploads'


# - - - FLASK INITIATION - - -
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # max upload size 1MB
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']


# - - - ERROR HANDLERS - - -
@app.errorhandler(400)
def wrong_file(e):
    return render_template('400.html')


@app.errorhandler(413)
def big_file(e):
    return render_template('413.html')


# - - - MAIN - - -
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/smile', methods=['POST', 'GET'])
def smile():
    if request.method == 'POST':
        return get_result(file=request.files['file'], model="smile")
    else:
        return render_template('upload.html')


@app.route('/face', methods=['POST', 'GET'])
def face():
    if request.method == 'POST':
        return get_result(file=request.files['file'], model="face")
    else:
        return render_template('upload.html')


def get_result(file, model):
    """
    takes uploaded file and model type of either "face" or "smile" as parameters, returns result template
    """
    uploaded_file = file
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(f"{UPLOAD_FOLDER}/{filename}")
    pic_path = f"{UPLOAD_FOLDER}/{filename}"
    if model == "smile":
        eval_result = eval.eval_smile(pic_path)
    if model == "face":
        eval_result = eval.eval_face(pic_path)
    return render_template('result.html', result=eval_result, pic=pic_path)


@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')


# - - - TO TEST LOCALLY - - -
# if __name__ == "__main__":
#     from waitress import serve
#     print("localhost:8080")
#     serve(app, host='0.0.0.0', port=8080)


if __name__ == "__main__":
    app.run(debug=True)
