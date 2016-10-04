from __future__ import absolute_import, division, print_function

import setproctitle; setproctitle.setproctitle('demo at ECCV 2016')

from flask import Flask, request, redirect, url_for, jsonify, send_from_directory
from time import time
import hashlib
import base64
from io import BytesIO
import numpy as np
import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import segmentation_demo as demo
import skimage.io
from skimage.transform import resize

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG'])
UPLOAD_FOLDER = './uploads/'
VIZ_FOLDER = './viz/'
PORT = sys.argv[2]

# global variables
app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
feature_cache = {}

# helpers
def setup():
    # uploads
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(VIZ_FOLDER):
        os.makedirs(VIZ_FOLDER)
    print('Finished setup')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def downsample_image(im, max_size=512):
    # calculate the resize scaling factor
    im_h, im_w = im.shape[:2]
    # make the long size no bigger than max_size
    scale = min(max_size/im_h, max_size/im_w)

    # resize and process the image
    new_h, new_w = int(scale*im_h), int(scale*im_w)
    im_resized = skimage.transform.resize(im, [new_h, new_w])

    return im_resized

# routes
@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('demo.html')

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file was uploaded.'})
    if allowed_file(file.filename):
        start = time()
        file_hash = hashlib.md5(file.read()).hexdigest()
        if file_hash in feature_cache:
            json = {'img_id': file_hash, 'time': time() - start}
            return jsonify(json)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file_hash + '.jpg')

        file.seek(0)
        im = skimage.io.imread(file)
        im = downsample_image(im)
        skimage.io.imsave(save_path, im)

        feature_cache[file_hash] = None
        json = {'img_id': file_hash, 'time': time() - start}
        return jsonify(json)
    else:
        return jsonify({'error': 'Please upload a JPG or PNG.'})

@app.route('/api/capture_image', methods=['POST'])
def capture_image():
    raw_img_string = request.form['raw_img_string']
    if len(raw_img_string) > 0:
        img_string = raw_img_string.split(',', 2)[1]
        img_data = base64.b64decode(img_string)
        start = time()
        file_hash = hashlib.md5(img_data).hexdigest()
        if file_hash in feature_cache:
            json = {'img_id': file_hash, 'time': time() - start}
            return jsonify(json)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file_hash + '.jpg')

        im = skimage.io.imread(BytesIO(img_data))
        im = downsample_image(im)
        skimage.io.imsave(save_path, im)

        feature_cache[file_hash] = None
        json = {'img_id': file_hash, 'time': time() - start}
        return jsonify(json)
    else:
        return jsonify({'error': 'No image captured through camera.'})

@app.route('/api/upload_question', methods=['POST'])
def upload_question():
    img_hash = request.form['img_id']
    question = request.form['question']
    if img_hash not in feature_cache:
        return jsonify({'error': 'Unknown image ID. Try uploading the image again.'})

    salt = str(time())
    img_ques_hash = hashlib.md5((img_hash + question + salt).encode('utf-8')).hexdigest()

    start = time()

    # attention visualization
    source_img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_hash + '.jpg')

    print('run demo start')
    path0 = os.path.join(VIZ_FOLDER, img_ques_hash + '.jpg')
    demo.run_demo(source_img_path, question, path0)
    print('run demo over')

    json = {'answer': question,
        'viz': [path0],
        'time': time() - start}
    return jsonify(json)

@app.route('/viz/<filename>')
def get_visualization(filename):
    return send_from_directory(VIZ_FOLDER, filename)

if __name__ == '__main__':
    setup()
    app.run(host='0.0.0.0', port=PORT, debug=False)
