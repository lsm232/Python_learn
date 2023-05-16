import tensorflow as tf
import random
import numpy as np
from tensorflow.contrib import rnn
from collections import Counter

def get_ch_lable(training_file):
    labels=''
    with open(training_file,'rb') as f:
        for label in f:
            labels=labels+label.decode('utf-8')
    return labels

def get_ch_lable_v(txt_file,word_num_map,txt_label=None):
    words_size=len(word_num_map)
    to_num=lambda word:word_num_map.get(word,print(word))
    if txt_file!=None:
        txt_label=get_ch_lable(txt_file)
    labels_vector=list(map(to_num,txt_label))
    return labels_vector


training_file='wordstest.txt'
training_data=get_ch_lable(training_file)
counter=Counter(training_data)
words=sorted(counter)
words_size=len(words)
word_num_map=dict(zip(words,range(words_size)))
wordlabel=get_ch_lable_v(training_file,word_num_map)

lr_rate=0.001
training_iters=10000
display_step=1000
n_input=4
n_hidden1=256
n_hidden2=512
n_hidden3=512


x=tf.placeholder('float',[None,n_input,1])
x1=tf.reshape(x,[-1,n_input])
x2=tf.split(x1,n_input,1)
wordy=tf.placeholder('float',[None,words_size])
rnn_cell=rnn.MultiRNNCell([rnn.LSTMCell(n_hidden1),rnn.LSTMCell(n_hidden2),rnn.LSTMCell(n_hidden3)])
outputs,states=rnn.static_rnn(rnn_cell,x2,dtype=tf.float32)
pred=tf.contrib.layers.fully_connected(outputs[-1],words_size)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=wordy))
optimizer=tf.train.AdamOptimizer(lr_rate).minimize(loss)
corrects=tf.equal(tf.argmax(pred,1),tf.argmax(wordy,1))
accuracy=tf.reduce_mean(tf.cast(corrects,dtype=tf.float32))

save_dir='./'
saver=tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step=0
    offset=random.randint(0,n_input+1)
    end_offset=n_input+1
    acc_total=0
    loss_total=0

    kpt=tf.train.latest_checkpoint(save_dir)
    print('kpt:',kpt)
    startepo=0
    if kpt!=None:
        saver.restore(sess,kpt)
        ind=kpt.find('-')
        startepo=int(kpt[ind+1:])
        print(startepo)
        step=startepo
    while step<training_iters:
        if offset > (len(training_data) - end_offset):
            offset = random.randint(0, n_input + 1)
        inwords=[[wordlabel[i]] for i in range(offset,offset+n_input)]
        inwords=np.reshape(np.array(inwords),[-1,n_input,1])
        out_onehot=np.zeros([words_size],dtype=float)
        out_onehot[wordlabel[offset+n_input]]=1.0
        out_onehot=np.reshape(out_onehot,[1,-1])

        _,accs,lossval,onehot_pred=sess.run([optimizer,accuracy,loss,pred],feed_dict={x:inwords,wordy:out_onehot})
        loss_total+=lossval
        acc_total+=accs
        if(step+1)%display_step==0:
            print('iter='+str(step+1)+',avg loss='+'{:.2f}%'.format(100*acc_total/display_step))
            acc_total=0
            loss_total=0
            in2=[words[wordlabel[i]] for i in range(offset,offset+n_input)]
            out2=words[wordlabel[offset+n_input]]
            out_pred=words[int(tf.argmax(onehot_pred,1).eval())]
            print('%s-[%s]vs[%s]'%(in2,out2,out_pred))

        step+=1
        offset+=(n_input+1)
    print('ok')
    saver.save(sess, save_dir + 'lsm.cpkt', global_step=step)

    while True:
        prompt='请输入%d个字:'%n_input
        sentence=input(prompt)
        inputword=sentence.strip()
        try:
            inputword=get_ch_lable_v(None,word_num_map,inputword)
            for i in range(32):
                keys=np.reshape(np.array(inputword),[-1,n_input,1])
                onehot_pred=sess.run(pred,feed_dict={x:keys})
                onehot_pred_index=int(tf.argmax(onehot_pred,1).eval())
                sentence="%s%s"%(sentence,words[onehot_pred_index])
                inputword=inputword[1:]
                inputword.append(onehot_pred_index)
            print(sentence)
        except:
            print('nono')







        c=1

c=1
