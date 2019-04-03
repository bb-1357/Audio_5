import tensorflow as tf

INPUT_NODE = 16000
OUTPUT_NODE = 1
LAYER1_NODE = 512
LAYER2_NODE = 256
LAYER3_NODE = 64


def forward(x,regularizer):
  w1 = get_weight([INPUT_NODE, LAYER1_NODE],regularizer)
  b1 = get_bias([LAYER1_NODE])
  y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
    
  w2 = get_weight([LAYER1_NODE, LAYER2_NODE],regularizer)
  b2 = get_bias([LAYER2_NODE])
  y2 = tf.matmul(y1,w2)+b2 

  w3 = get_weight([LAYER2_NODE, LAYER3_NODE],regularizer)
  b3 = get_bias([LAYER3_NODE])
  y3 = tf.matmul(y2,w3)+b3

  w4 = get_weight([LAYER3_NODE, OUTPUT_NODE],regularizer)
  b4 = get_bias([OUTPUT_NODE])
  y = tf.matmul(y3,w4)+b4
      
  return y

def get_weight(shape,regularizer):
  w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
  if regularizer !=None:tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
  return w

def get_bias(shape):
  b = tf.Variable(tf.zeros(shape))
  return b
