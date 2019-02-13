#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:31:17 2019

@author: anaclaudia
"""

import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt
from model_low_sparse import Model


  
def buildRandomBoolDataSet(Nsamples=2000,Dinput=1000,seed=0):
    np.random.seed(seed)
    features, labels = (np.random.randint(2,size=(Nsamples,Dinput)), np.eye(Nsamples))
    features = (np.float32(features)*2-1)/np.sqrt(Dinput)
#    labels = (np.float32(features)*2-1)/np.sqrt(Doutput)
    labels = np.float32(labels)
    dataset = tf.data.Dataset.from_tensor_slices((features,labels))
    return dataset
  
sess = tf.InteractiveSession()
input_dim = 100
#num_samples = 100
#sparseness = 0.9
learning_rate = 0.1
model_seed = 0
data_seed = 0
nsteps = 10
max_count = 10


#sparse = np.linspace(0.0,0.9,10)
#sparse = np.linspace(0.0,0.4,4)
sparse = np.arange(0.0, 0.4, 0.1)


#num_samplesV = np.power(10,np.arange(0.2,4.1,0.2))
#num_samplesV = (np.power(10,np.arange(0.2,4.1,0.2)))
num_samplesV = np.arange(1,4)

#data_seedV = np.arange(20)
data_seedV = np.arange(3)

acc = np.ones((len(sparse),len(num_samplesV),len(data_seedV))) * np.nan

for zz,sparseness in enumerate(sparse):
    for ii,num_samples in enumerate(num_samplesV):
        for jj,data_seed in enumerate(data_seedV):
                
                myDataSet = buildRandomBoolDataSet(num_samples.astype(np.int64),input_dim, data_seed).batch(num_samples)                
                myIter = myDataSet.make_initializable_iterator()
                myFeatures, myLabels = myIter.get_next()               
                x = myFeatures
                y_ = myLabels              
                #model = Model(x, y_, spars, 0.1) # simple 2-layer network
                print("data_seed")
                print(data_seed)

                model = Model(x, y_, sparseness, learning_rate, seed=data_seed) # simple 2-layer network
                model.set_vanilla_loss()              
                # initialize variables
                sess.run(tf.global_variables_initializer())    
                sess.run(myIter.initializer)
                #acc[0] = model.accuracy.eval()            
                count  = 1            
                for ss in np.arange(1,nsteps+1):
                    sess.run(myIter.initializer)
                    model.train_step.run()
                    sess.run(myIter.initializer)
                    #acc[ss] = model.accuracy.eval()
                    
                    acc[zz,ii,jj] = model.accuracy.eval()
                    
sess.close()
 
# %%

ax = plt.subplot(111)
#plt.plot(acc[:,:,0])
for zz,sparseness in enumerate(sparse):  
    plt.plot(num_samplesV,np.median(acc[zz,:,:],axis=1), label="spars=%1.2f"%(sparseness,))
#    plt.semilogx(num_samplesV,np.percentile(acc[zz,:,:],25,axis=1))
#    plt.semilogx(num_samplesV,np.percentile(acc[zz,:,:],75,axis=1))

plt.xlabel('NSamples ')
plt.ylabel('Accuracy')
plt.grid(True)
#plt.axis([5,9, 0.0,1.0])
leg = plt.legend(bbox_to_anchor=(1.1, 1.05))
leg.get_frame().set_alpha(0.5)
plt.show()

plt.savefig('accuracy.eps')
   