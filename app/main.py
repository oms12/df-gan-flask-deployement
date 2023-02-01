from flask import Flask, request, jsonify
from flask import render_template, Response
import os
from torch_utils import sample

UPLOAD_FOLDER='app/static/'
app = Flask(__name__, template_folder='./templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
# @app.route('/generate', methods = ['POST'])
# def generate():
#     # get the text
#     caption = request.json['caption']
#     # convert to tensor
#     captionList = [caption]
#     # generation of the image
#     # return the( image json
    
#     return sample(captionList)

 
@app.route('/')
def index():
    return render_template('index.html')
    

@app.route('/',methods=['POST','GET'])
def tasks():
    if request.method == 'POST':
        input_text = request.form['text']
        captionList = [input_text]
        sample(captionList)
                              
    return render_template('index.html')
# @app.route('/')
# @app.route('/index')
# def show_index():
#     full_filename = os.path.join(app.config['UPLOAD_FOLDER'],'shovon.jpg')
#     return render_template("index.html", user_image = full_filename)
