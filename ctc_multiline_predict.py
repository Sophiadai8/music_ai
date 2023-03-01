import argparse
import tensorflow as tf
import ctc_utils
import cv2
import numpy as np
from pythonosc import osc_message_builder
from pythonosc import udp_client
import time

show_image = True

parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
parser.add_argument('-image',  dest='image', type=str, required=True, help='Path to the input image.')
parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
parser.add_argument('-vocabulary', dest='voc_file', type=str, required=True, help='Path to the vocabulary file.')
parser.add_argument('-lines', dest='lines', type=int, required=False, help='Lines in a sheet.')
args = parser.parse_args()

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Read the dictionary
dict_file = open(args.voc_file,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()

def detect_line(data):
    total = len(data)
    dark = 0
    for d in data:
        if d < 220:
            dark +=1 
    if dark / total > 0.6:
        return True
    else:
        return False


# Restore weights
saver = tf.train.import_meta_graph(args.model)
saver.restore(sess,args.model[:-5])

graph = tf.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)    


def decode_image(image):

    image = ctc_utils.resize(image, HEIGHT)
    
    image = ctc_utils.normalize(image)
    if show_image:
        cv2.imshow("original", image)
        cv2.waitKey()     
    image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

    seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

    prediction = sess.run(decoded,
                        feed_dict={
                            input: image,
                            seq_len: seq_lengths,
                            rnn_keep_prob: 1.0,
                        })

    return ctc_utils.sparse_tensor_to_strs(prediction)

img = cv2.imread('Data/Example/3_1_blue.png', False)
print("my image", img.shape) # Print image shape
height, width = img.shape
# cv2.imshow("original", img)


i = 0
lines = []
for data in img:
    isLine = detect_line(data)
    i += 1
    # print (i, isLine)
    if isLine:
        lines.append(i)

print(lines)

lines_cleaned = []
for n in range(len(lines)-1):
    if lines[n+1] != lines[n] + 1:
        lines_cleaned.append(lines[n])
    
lines_cleaned.append(lines[-1])
print(lines)
print (lines_cleaned)
lines = lines_cleaned


str_predictions = []

# x = cv2.imread(args.image,False)
# print ("x", x.shape)
# str_predictions.append(decode_image(x)[0])

bar_gap = int((lines[4]-lines[0])/4*2.5)

for c in range(int(len(lines)/5)):
    # Cropping an image
    start = lines[5*c]-bar_gap
    stop = lines[5*c+4 ]+bar_gap
    if start < 0:
        start = 0
    if stop > height:
        stop = height
    print ("cropped range:", start, stop)
    cropped_image = img[start:stop, 0:width]
    if show_image:
        cv2.imshow("original", cropped_image)
        cv2.waitKey()  
    # x = cv2.imread(args.image,False)
    str_predictions.append(decode_image(cropped_image)[0])
    if len(str_predictions) >= 2:
        str_predictions[0] += str_predictions[len(str_predictions)-1]

sender = udp_client.SimpleUDPClient('127.0.0.1', 51241)

# str_predictions = ["clef-C1", "keySignature-EbM", "timeSignature-2/4", "multirest-23", "barline", "rest-quarter", "rest-eighth", "note-Bb4_eighth", "barline",
#                    "note-Bb4_quarter", "note-G4_eighth", "barline", "note-Eb5_quarter", "note-D5_eighth", "barline", "note-C5_eighth", "note-C5_eighth", "rest-quarter"]

j = 0
def note_length(l, i):
    global j
    j +=1
    print ("note_name", l[0], "note_length", l[-1])
    if l[i] == "eighth":
        return 1/2
    elif l[i] == "eighth.":
        return 3/4
    elif l[i] == "quarter":
        return 1
    elif l[i] == "quarter.":
        return 1.5
    elif l[i] == "half":
        return 2
    elif l[i] == "half.":
        return 2.5
    elif l[i] == "whole":
        return 4
    elif l[i] == "whole.":
        return 6
    elif l[i] == "sixteenth":
        return 1/4
    elif l[i] == "sixteenth.":
        return 3/8
    elif l[i] == "thirty":
        if l[i+1] == "second.":
            return 3/16
        else:
            return 1/8
    elif l[i] == "double":
        if l[i+1] == "whole.":
            return 12
        else:
            return 8
    elif l[i] == "quadruple":
        if l[i+1] == "whole.":
            return 24
        else:
            return 16

for w in str_predictions[0]:
    print(int2word[w]),
    print('\t'),

for w in str_predictions[0]:
    w = int2word[w].split("-")
    if w[0] == "clef" or w[0] == "keySignature" or w[0] == "timeSignature" or w[0] == "barline" or w[0] == "multirest":
        continue
    # if w[0] == "multirest":
    #     sl = int(w[-1])
    #     am = 0
    #     pl = "C"
    #     su = int(w[-1])
    if w[0] == "rest":
        # m = w[1].split("_")
        am = 0
        pl = "C"
        su = note_length(w, 1)
    elif w[0] == "note" or w[0] == "gracenote":
        m = w[1].split("_")
        note = list(m[0])
        # if note[0] == "C" and note[1] == "#":
        #     note[2] = str(int(note[2]) +1)
        #     # note[2] = str(note[2])
        #     note[1] = "s"
        #     note[0:3] = [''.join(note[0:2])]
        #     pl = note[0]
        if note[1] == "#":
            note[1] = "s"
            note[0:3] = [''.join(note[0:3])]
            pl = note[0]
        else:
            pl = m[0]
        am = 1
        su = note_length(m, 1)
        
    # su = su/1.3
    am = am*9
    # print("send", pl, su, am, "piano")
    sender.send_message('/sci/thing', [pl, su, am, "piano"])
    # sender.send_message('/sci/thing', ['C', 1, 3, "piano"])
    time.sleep(su)
print("total musical notations decoded:", j)