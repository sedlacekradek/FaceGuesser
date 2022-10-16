from flask import Flask, render_template, request, abort
import os
from werkzeug.utils import secure_filename
import smile_eval
import face_eval


# - - - CONSTANTS - - -
UPLOAD_FOLDER = 'static/uploads'


# - - - FLASK INITIATION - - -
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # max upload size 1mb
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']


# - - - ERROR HANDLERS - - -
@app.errorhandler(400)
def wrong_file():
    return render_template('400.html')


# - - - MAIN - - -
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/smile', methods=['POST', 'GET'])
def smile():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                abort(400)
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        pic_path = f"static/uploads/{filename}"
        result = smile_eval.eval_smile(pic_path)
        return get_result(pic_result=result, pic_path=pic_path)
    else:
        return render_template('upload.html')


@app.route('/face', methods=['POST', 'GET'])
def face():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                abort(400)
            uploaded_file.save(f"{UPLOAD_FOLDER}/{filename}")
        pic_path = f"{UPLOAD_FOLDER}/{filename}"
        result = face_eval.eval_face(pic_path)
        return get_result(pic_result=result, pic_path=pic_path)
    else:
        return render_template('upload.html')


def get_result(pic_result, pic_path):
    return render_template('result.html', result=pic_result, pic=pic_path)


@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')



# - - - TO TEST LOCALLY - - -
# if __name__ == "__main__":
#     from waitress import serve
#     print("localhost:8080")
#     serve(app, host='0.0.0.0', port=8080)


# if __name__ == "__main__":
#     app.run(debug=True)
