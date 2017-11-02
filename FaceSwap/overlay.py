import dlib
import cv2
import numpy as np
import os
import random
import glob

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *
from random import *
import FaceRendering
import utils
from werkzeug.utils import secure_filename

import pygame
import smtplib



import os
from flask import Flask,abort,render_template,request,redirect,url_for, send_from_directory
from subprocess import call

app = Flask(__name__)
#app.run(host='0.0.0.0', port=5432)

UPLOAD_FOLDER = '../data/image/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

@app.route('/')
def index():
    return "webserver running"

@app.route('/upload',methods = ['GET','POST'])
def upload_file():
    if request.method =='POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            return ProcessImage(filename)



def ProcessImage(filename):
 
 #you need to download shape_predictor_68_face_landmarks.dat from the link below and unpack it where the solution file is
#http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

#loading the keypoint detection model, the image and the 3D model
    predictor_path = "../shape_predictor_68_face_landmarks.dat"
    file_list = glob.glob("../data/image/*.jpg")
    image_index = randint(0, len(file_list)-1)
    image_name = file_list[image_index]
    source_image_name = "../data/image/uploads/"+filename

    #the smaller this value gets the faster the detection will work
    #if it is too small, the user's face might not be detected
    maxImageSizeForDetection = 150

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("../candide.npz")

    projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

    modelParams = None
    lockedTranslation = True
    drawOverlay = False
    writer = None


    sourceImg = cv2.imread(source_image_name)
    textureImg = cv2.imread(image_name)
    textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)


    renderer = FaceRendering.FaceRenderer(sourceImg, textureImg, textureCoords, mesh)

    while True:
        sourceImg = cv2.imread(source_image_name)
        shapes2D = utils.getFaceKeypoints(sourceImg, detector, predictor, maxImageSizeForDetection)

        if shapes2D is not None:
            for shape2D in shapes2D:
                #3D model parameter initialization
                modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

                #3D model parameter optimization
                modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

                #rendering the model to an image
                shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
                renderedImg = renderer.render(shape3D)

                #blending of the rendered face with the image
                mask = np.copy(renderedImg[:, :, 0])
                renderedImg = ImageProcessing.colorTransfer(sourceImg, renderedImg, mask)
                sourceImg = ImageProcessing.blendImages(renderedImg, sourceImg, mask)
           

                #drawing of the mesh and keypoints
                if drawOverlay:
                    drawPoints(sourceImg, shape2D.T)
                    drawProjectedShape(sourceImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)


        cv2.imwrite( "../data/image/completed/"+filename, sourceImg );
        #sendEmail(filename)
        pygame.quit()

        return send_from_directory(os.path.join("../data/image/completed/"), filename) 






if __name__ == '__main__':
    #app.run(debug=False)
    app.run(host="0.0.0.0")


