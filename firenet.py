

import cv2
import os
import sys
import math


import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression


def construct_firenet (x,y, training=False):


    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

    network = conv_2d(network, 64, 5, strides=4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 128, 4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 1, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    if(training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    if(training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')


    if(training):
        network = regression(network, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)


    model = tflearn.DNN(network, checkpoint_path='firenet',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model


if __name__ == '__main__':


    model = construct_firenet (224, 224, training=False)
    print("Constructed FireNet ...")

    model.load(os.path.join("models/FireNet", "firenet"),weights_only=True)
    print("Loaded CNN network weights ...")


    rows = 224
    cols = 224


    windowName = "Live Fire Detection - FireNet CNN";
    keepProcessing = True;

    if len(sys.argv) == 2:


        video = cv2.VideoCapture(sys.argv[1])
        print("Loaded video ...")


        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);


        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_time = round(1000/fps);

        while (keepProcessing):

            start_t = cv2.getTickCount();


            ret, frame = video.read()
            if not ret:
                print("... end of video file reached");
                break;


            small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)


            output = model.predict([small_frame])


            if round(output[0][0]) == 1:
                cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
                cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
            else:
                cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
                cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);

            stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

            cv2.imshow(windowName, frame);

            key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
            if (key == ord('x')):
                keepProcessing = False;
            elif (key == ord('f')):
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    else:
        print("usage: python firenet.py videofile.ext");

