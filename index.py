from turtle import width
import cv2
import numpy as np
from flask import render_template
from flask import Flask
from flask import request
from PIL import Image
from skimage import io

app = Flask(__name__)

@app.route('/')
@app.route('/index/', methods=['POST','GET'])
def index():
    if(request.method == 'POST'):        
        img = request.files['img']
        img.save('static/' + img.filename)

        return render_template('index.html', img=img.filename)

    return render_template('index.html')

# 1 Extract BGR
@app.route('/index/extract-BGR/', methods=['POST', 'GET'])
def extractBGR():
    if (request.method == 'POST' and request.form['img']):
        extract = getBGRPixels(request.form['img'])
        return render_template('index.html', img=request.form['img'], extract=extract)
    
    return render_template('index.html')

# 2 Grayscale Constanta
@app.route('/index/grayscale/', methods=['POST', 'GET'])
def grayscale():
    img_name = request.form['img']  
    img_file = cv2.imread('static/' + img_name)
    path = 'static/grayscale_' + img_name
    grayscale_filename = 'grayscale_' + img_name

    if (request.method == 'POST' and img_name):
        grayscale = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path, grayscale)

        return render_template('index.html', img=request.form['img'], grayscale=grayscale_filename)

    return render_template('index.html')
    
# 3 Grayscale Average
@app.route('/index/grayscale-avg/', methods=['POST', 'GET'])
def grayscaleAvg():
    img_name = request.form['img']  
    img_file = cv2.imread('static/' + img_name)
    path = 'static/grayscale_avg_' + img_name
    grayscale_filename = 'grayscale_avg_' + img_name

    if (request.method == 'POST' and img_name):
        R, G, B = img_file[:, :, 0], img_file[:, :, 1], img_file[:, :, 2]
        avg = (R + G + B) / 3
        grayscale = avg + avg + avg
        cv2.imwrite(path, grayscale)

        return render_template('index.html', img=request.form['img'], grayscale_avg=grayscale_filename)

    return render_template('index.html')

# 4 Detect Shapes
@app.route('/index/detect-shapes/', methods=['POST', 'GET'])
def capture():
    img_name = request.form['img']  
    path = 'static/capture_' + img_name
    shapes_filename = 'capture_' + img_name

    if (request.method == 'POST' and img_name):
        img_shapes = detectCountour(img_name)
        cv2.imwrite(path, img_shapes)

        return render_template('index.html', img=request.form['img'], shapes=shapes_filename)

    return render_template('index.html')

# 6 Inverse
@app.route('/index/inverse/', methods=['POST', 'GET'])
def inverse():
    img_name = request.form['img']  
    path = 'static/inverse_' + img_name
    inverse_filename = 'inverse_' + img_name

    if (request.method == 'POST' and img_name):
        img = cv2.imread('static/' + img_name)
        img_inverse = 255 - img
        cv2.imwrite(path, img_inverse)

        return render_template('index.html', img=request.form['img'], inverse=inverse_filename)

    return render_template('index.html')

# 7 Crop
@app.route('/index/crop/', methods=['POST', 'GET'])
def crop():
    img_name = request.form['img']  
    path = 'static/crop_' + img_name
    crop_filename = 'crop_' + img_name

    width1 = request.form['widthFrom']
    width2 = request.form['widthEnd']
    height1 = request.form['heightFrom']
    height2 = request.form['heightEnd']

    if(width1 > width2 or height1 > height2):
        message = 'Nilai From Harus lebih kecil Dari End!'
        return render_template('index.html', message=message)
    
    img = Image.open('static/' + img_name)
    width, height = img.size
    img_arr = np.array(img)

    width1 = int(width * (int(width1)/100))
    width2 = int(width * (int(width2)/100))
    height1 = int(height * (int(height1)/100))
    height2 = int(height * (int(height2)/100))

    img_arr = img_arr[width1:width2, height1:height2]
    img_crop = Image.fromarray(img_arr)
    img_crop.save(path)

    return render_template('index.html', img=request.form['img'], crop=crop_filename)

# 8 Brightness add CV
@app.route('/index/brightness-add-cv/', methods=['POST', 'GET'])
def brightnessAddCV():
    img_name = request.form['img']  
    path = 'static/brightness_addCV_' + img_name
    brightness_filename = 'brightness_addCV_' + img_name
    image = io.imread('static/' + img_name)
    
    new_image = cv2.add(image, 100)
    cv2.imwrite(path, new_image)

    return render_template('index.html', img=request.form['img'], brightness_cv_add=brightness_filename)

# 9 Brightness add
@app.route('/index/brightness-add/', methods=['POST', 'GET'])
def brightnessAdd():
    img_name = request.form['img']  
    path = 'static/brightness_add_' + img_name
    brightness_filename = 'brightness_add_' + img_name
    image = io.imread('static/' + img_name)
    
    image = np.asarray(image).astype('uint16')
    image = image+100
    image = np.clip(image, 0, 255)
    new_image = image.astype('uint8')
    new_image = Image.fromarray(new_image)
    new_image.save(path)

    return render_template('index.html', img=request.form['img'], brightness_add=brightness_filename)

# 10 Brightness subt CV
@app.route('/index/brightness-subt-cv/', methods=['POST', 'GET'])
def brightnessSubtCV():
    img_name = request.form['img']  
    path = 'static/brightness_subtCV_' + img_name
    brightness_filename = 'brightness_subtCV_' + img_name
    image = io.imread('static/' + img_name)
    
    new_image = cv2.subtract(image, 100)
    new_image = np.clip(new_image, 0, 255)
    cv2.imwrite(path, new_image)

    return render_template('index.html', img=request.form['img'], brightness_subt_cv=brightness_filename)

# 11 Brightness subt CV
@app.route('/index/brightness-subt/', methods=['POST', 'GET'])
def brightnessSubt():
    img_name = request.form['img']  
    path = 'static/brightness_subt_' + img_name
    brightness_filename = 'brightness_subt_' + img_name
    image = io.imread('static/' + img_name)

    image = image.astype('uint16')
    image = image-100
    image = np.clip(image, 0, 255)
    image = image.astype('uint8')
    image = Image.fromarray(image)
    image.save(path)

    return render_template('index.html', img=request.form['img'], brightness_subt=brightness_filename)

# 12 Brightness Mult CV
@app.route('/index/brightness-mult-cv/', methods=['POST', 'GET'])
def brightnessMultCV():
    img_name = request.form['img']  
    path = 'static/brightness_multCV_' + img_name
    brightness_filename = 'brightness_multCV_' + img_name
    image = io.imread('static/' + img_name)
    
    new_image = cv2.multiply(image, 0.5)
    new_image = np.clip(new_image, 0, 255)
    cv2.imwrite(path, new_image)

    return render_template('index.html', img=request.form['img'], brightness_mult_cv=brightness_filename)

# 13 Brightness Mult
@app.route('/index/brightness-mult/', methods=['POST', 'GET'])
def brightnessMult():
    img_name = request.form['img']  
    path = 'static/brightness_mult_' + img_name
    brightness_filename = 'brightness_mult_' + img_name
    image = io.imread('static/' + img_name)

    image = image.astype('uint16')
    image = image*1.25
    image = np.clip(image, 0, 255)
    image = image.astype('uint8')
    image = Image.fromarray(image)
    image.save(path)

    return render_template('index.html', img=request.form['img'], brightness_mult=brightness_filename)

# 14 Brightness Div CV
@app.route('/index/brightness-div-cv/', methods=['POST', 'GET'])
def brightnessDivCV():
    img_name = request.form['img']  
    path = 'static/brightness_divCV_' + img_name
    brightness_filename = 'brightness_divCV_' + img_name
    image = io.imread('static/' + img_name)
    
    new_image = cv2.divide(image, 1)
    new_image = np.clip(new_image, 0, 255)
    cv2.imwrite(path, new_image)

    return render_template('index.html', img=request.form['img'], brightness_div_cv=brightness_filename)

# 15 Brightness Div
@app.route('/index/brightness-div/', methods=['POST', 'GET'])
def brightnessDiv():
    img_name = request.form['img']  
    path = 'static/brightness_div_' + img_name
    brightness_filename = 'brightness_div_' + img_name
    image = io.imread('static/' + img_name)

    image = np.asarray(image).astype('uint16')
    image = image/2
    image = np.clip(image, 0, 255)
    image = image.astype('uint8')
    image = Image.fromarray(image)
    image.save(path)

    return render_template('index.html', img=request.form['img'], brightness_div=brightness_filename)

# 16 Bitwise And
@app.route('/index/bitwise-and/', methods=['POST', 'GET'])
def bitwiseAnd():
    
    img_name = request.form['img']  
    img_name2 = request.files['img2']  

    path = 'static/bitwise_and_' + img_name
    bitwise_filename = 'bitwise_and_' + img_name
    image = io.imread('static/' + img_name)
    image2 = io.imread('static/' + img_name2.filename)

    if(image.shape != image2.shape):
        return render_template('index.html', message='Dimensi image harus sama!')


    bit_and = cv2.bitwise_and(image,image2)
    cv2.imwrite(path,bit_and)

    return render_template('index.html', img=request.form['img'], img2=request.files['img2'].filename, bitwise_and=bitwise_filename)

# 17 Bitwise Or
@app.route('/index/bitwise-or/', methods=['POST', 'GET'])
def bitwiseOr():

    img_name = request.form['img']
    img_name2 = request.files['img2']

    path = 'static/bitwise_or_' + img_name
    bitwise_filename = 'bitwise_or_' + img_name
    image = io.imread('static/' + img_name)
    image2 = io.imread('static/' + img_name2.filename)

    if (image.shape != image2.shape):
        return render_template('index.html', message='Dimensi image harus sama!')

    bit_or = cv2.bitwise_or(image, image2)
    cv2.imwrite(path, bit_or)

    return render_template('index.html', img=request.form['img'], img2=request.files['img2'].filename, bitwise_or=bitwise_filename)

# 18 Bitwise Not
@app.route('/index/bitwise-not/', methods=['POST', 'GET'])
def bitwiseNot():

    img_name = request.form['img']

    path = 'static/bitwise_not_' + img_name
    bitwise_filename = 'bitwise_not_' + img_name
    image = io.imread('static/' + img_name)

    bit_not = cv2.bitwise_not(image)
    cv2.imwrite(path, bit_not)

    return render_template('index.html', img=request.form['img'], bitwise_not=bitwise_filename)

# 19 Bitwise XOr
@app.route('/index/bitwise-xor/', methods=['POST', 'GET'])
def bitwiseXor():

    img_name = request.form['img']
    img_name2 = request.files['img2']

    path = 'static/bitwise_xor_' + img_name
    bitwise_filename = 'bitwise_xor_' + img_name
    image = io.imread('static/' + img_name)
    image2 = io.imread('static/' + img_name2.filename)

    if (image.shape != image2.shape):
        return render_template('index.html', message='Dimensi image harus sama!')

    bit_xor = cv2.bitwise_xor(image, image2)
    cv2.imwrite(path, bit_xor)

    return render_template('index.html', img=request.form['img'], img2=request.files['img2'].filename, bitwise_xor=bitwise_filename)


# Detect Countour
def detectCountour(src):
    # reading image
    img = cv2.imread('static/' + src)

    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # using a findContours() function
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    for contour in contours:
    # here we are ignoring first counter because
    # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)

        # using drawContours() function
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

    return img

# Scale Down by width
def scaleDown(img, scale):
    width_scale = img.shape[1] - scale
    height = img.shape[0]
    print(img.shape[1], height)
    print(scale, height-width_scale-scale)
    return cv2.resize(img, (scale, height-width_scale), interpolation=cv2.INTER_LINEAR)

# BGR
def getBGRPixels(src):
    img = cv2.imread('static/' + src)
    # img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_LINEAR)
    img = scaleDown(img, 10)

    rows, cols, _ = img.shape
    result = ''

    for i in range(rows):
        for j in range(cols):
            k = img[i, j]
            result += str(k)

    return result









if __name__ == '__main__':
   app.run(debug=True)


