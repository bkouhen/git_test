# _*_coding:utf-8_*_
import tensorflow as tf, sys
import cv2
import numpy as np

cv2.namedWindow('Take a phtoto')

capture = cv2.VideoCapture(0)
_, frame = capture.read()
print(frame)
while frame is not None:
    cv2.imshow('Take a phtoto', frame)

    key = cv2.waitKey(10)
    if key == ord('s'):
        cv2.imwrite('screenshot.jpg', frame)
    elif key == ord('q'):
        break

    _, frame = capture.read()
    
# Read in the image_data
image_data = tf.gfile.FastGFile('screenshot.jpg', 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels_smartphones.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph_smartphones.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
