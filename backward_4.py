import tensorflow as tf
import forward_4 as forward
import os
import scipy.io.wavfile as wave
from python_speech_features import mfcc
import numpy as np
import wave
import matplotlib
BATCH_SIZE = 20
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 100000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'audio_model'
num_examples=10000  
a=[1]
b=[2]
def get_wav_mfcc(wav_path):
    f = wave.open(wav_path,'rb')
    params = f.getparams()
    nchannels,sampwidth,framerate,nframes = params[:4]
    strData = f.readframes(nframes) 
    waveData = np.fromstring(strData,dtype=np.int16) 
    waveData = waveData*1.0/(max(abs(waveData)))
    waveData = np.reshape(waveData,[nframes,nchannels]).T
    f.close()
    
    data = list(np.array(waveData[0]))
    while len(data)>16000:
        del data[len(wavData[0])-1]
        del data[0]
    while len(data)<16000:
        data.append(0)


    data = np.array(data)
    data = data **2
    data = data **0.5
    return data


def create_datasets():
    wavs=[] 
    labels=[] 
    testwavs=[]
    testlabels=[]
    labsInd=[]      
    testlabsInd=[]  

    path="/home/hp209/Desktop/Audio/train/A2/"
    files = os.listdir(path)
    for i in files:
        #print(i)
        waveData = get_wav_mfcc(path+i)
        #print(waveData)
        wavs.append(waveData)
        if ("pure" in labsInd)==False:
            labsInd.append("pure")
        labels.append(a)

    path="/home/hp209/Desktop/Audio/train-noise/0db/cafe/"
    files = os.listdir(path)
    for i in files:
        #print(i)
        waveData = get_wav_mfcc(path+i)
        wavs.append(waveData)
        if ("cafe" in labsInd)==False:
            labsInd.append("cafe")
        labels.append(b)

    path="/home/hp209/Desktop/Audio/test/D6/"
    files = os.listdir(path)
    for i in files:
        
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("pure" in testlabsInd)==False:
            testlabsInd.append("pure")
        testlabels.append(testlabsInd.index("pure"))
    
    path="/home/hp209/Desktop/Audio/test-noise/0db/cafe/"
    files = os.listdir(path)
    for i in files:
        #print(i)
        waveData = get_wav_mfcc(path+i)
        testwavs.append(waveData)
        if ("cafe" in testlabsInd)==False:
            testlabsInd.append("cafe")
        testlabels.append(testlabsInd.index("cafe"))

    wavs=np.array(wavs)
    labels=np.array(labels)
    testwavs=np.array(testwavs)
    testlabels=np.array(testlabels)
    return (wavs,labels),(testwavs,testlabels),(labsInd,testlabsInd)
    
    


def backward(wavs,labels):
    x = tf.placeholder(tf.float32, [None ,forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None ,forward.OUTPUT_NODE])
    y = forward.forward(x, REGULARIZER)
    global_step= tf.Variable(0, trainable=False)
    
    #损失函数正则化
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem = tf.reduce_mean(ce)
    loss=cem+tf.add_n(tf.get_collection('losses'))

    #正则化，滑动平均学习率，滑动平均
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase = True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step) 
    #滑动平均ema
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name='train')
    #实例化saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        #初始化
        for i in range(STEPS):
            xs, ys = wavs,labels
            #how to do with batchsize?
            _, loss_value, step = sess.run([train_op, loss, global_step],feed_dict={x: xs,y_: ys})
            if i % 100== 0:
                print("After %d training step(s), loss on training batch is %f "%(step,loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        #主要修改的就是这个xs和ys部分，要求体现出标签和结果

def main():
    (wavs,labels),(testwavs,testlabels),(labsInd,testlabsInd)=create_datasets()
    backward(wavs,labels)

if __name__ == '__main__':
    main()
