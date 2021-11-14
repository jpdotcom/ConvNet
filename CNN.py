import numpy as np 
import cupy as cp
import tensorflow as tf
import time
from cupy.lib.stride_tricks import as_strided
import math
 
class CNN:
    def getrandparameters(self,size,is_filter):
      range=0
      if (is_filter):
        range=math.sqrt(3/(size[0]*size[1]*size[2]))
      else:
        range=math.sqrt(3/(size[1]))
     
      return cp.random.uniform(-1*range,range,size)
 
    def __init__(self,sizes,filter_sizes):
      #filters setup
    
      self.layer1_filters=self.getrandparameters((filter_sizes[0][0],filter_sizes[0][1],sizes[0],sizes[1]),1)
    
      self.layer2_filters=self.getrandparameters((filter_sizes[1][0],filter_sizes[1][1],sizes[1],sizes[2]),1)
      
      self.layer3_filters=self.getrandparameters((filter_sizes[2][0],filter_sizes[2][1],sizes[2],sizes[3]),1)
      


      #bias setup 
 
      self.CNN_layer1_bias=cp.zeros((sizes[1],))
 
      self.CNN_layer2_bias=cp.zeros((sizes[2],))
 
      self.CNN_layer3_bias=cp.zeros((sizes[3],))
 
      #MLP setup
 
 
      self.MLP_weights=self.getrandparameters((sizes[4],sizes[3]),0)
      self.MLP_bias=cp.zeros((sizes[4],1))

      #inverse pooling setup
 
      self.inverse_pool_map=[0]*3
    def conv2d(self,a, b):
      
      a=self.get_submatrices(a,b.shape)
      #return cp.einsum('abcijk,ijkd', a, b)
      return cp.tensordot(a, b, axes=3)
 
 
    def ReLU(self,x):
      return cp.maximum(x,0,x)
    
    def ReLUd(self,x):
      return 1*(x>0)
      
    
    def softmax(self,x):
    
        x2=cp.max(x,axis=1,keepdims=True)
        e_x = cp.exp(x - x2)
        e_sum=cp.sum(e_x,axis=1,keepdims=True)
        return e_x/e_sum
    
    def get_submatrices(self,layer,size,s=1):
      
      a=layer
      Hout = (a.shape[1] - size[0]) // s + 1
      Wout = (a.shape[2] - size[1]) // s + 1
      Stride = (a.strides[0], a.strides[1] * s, a.strides[2] * s, a.strides[1], a.strides[2], a.strides[3])
 
      a = as_strided(a, (a.shape[0], Hout, Wout, size[0], size[1], a.shape[3]), Stride)
      return a
    
    def conv2d_grad(self,a, b):
  
      a=self.get_submatrices_grad(a,b.shape)
      
      #return cp.einsum("iabjkc,ijkd",a,b)
      return cp.tensordot(a,b,axes=((0,3,4),(0,1,2)))
    def get_submatrices_grad(self,layer,size,s=1):
      
      a=layer
      Hout = (a.shape[1] - size[1]) // s + 1
      Wout = (a.shape[2] - size[2]) // s + 1
      Stride = (a.strides[0], a.strides[1] * s, a.strides[2] * s, a.strides[1], a.strides[2], a.strides[3])
    
      a = as_strided(a, (a.shape[0], Hout, Wout, size[1], size[2], a.shape[3]), Stride)
      return a
    
    def full_conv(self,a,b):

      fw=b.shape[0]-1

      fh=b.shape[1]-1
      
      a=cp.pad(a,((0,0),(fw,fw),(fh,fh),(0,0)))
    
      b=cp.rot90(cp.rot90(b))
      b=b.transpose(0,1,3,2)
      ans=self.conv2d(a,b)
      return ans
    
    def pooling(self,size,layer,idx):
        
        batch_size,h,w,channels=layer.shape
        m,n=size 
        hm=h//m 
        wn=w//n
        prev_layer=layer
        layer=layer.reshape(batch_size,hm,m,wn,n,channels)
        ans = cp.max(layer,axis=(2,4))
        check=ans.reshape(batch_size,hm,1,wn,1,channels)
        inv=(cp.equal(layer,check))*1
        inv=inv.reshape(batch_size,h,w,channels)
        self.inverse_pool_map[idx]=inv
        
        return ans
 
 
 
    def run(self,image,p,training):
      s=time.time()
    
      self.input=image
      batch_size,h,w,channels=self.input.shape
      if training:
            self.input_dropout=cp.random.binomial(1, p[0], size=self.input.shape) / p[0]
            self.input*=self.input_dropout
      
      self.CNN_layer1=self.ReLU(self.conv2d(self.input,self.layer1_filters)+self.CNN_layer1_bias)
     
      
      self.pool1=self.pooling((2,2),self.CNN_layer1,0)
 
      if training:
            self.pool1_dropout=cp.random.binomial(1, p[1], size=self.pool1.shape) / p[1]
            self.pool1*=self.pool1_dropout
     
      self.CNN_layer2=self.ReLU(self.conv2d(self.pool1,self.layer2_filters)+self.CNN_layer2_bias)
      self.pool2=self.pooling((2,2),self.CNN_layer2,1)
      
      if training:
            self.pool2_dropout=cp.random.binomial(1, p[2], size=self.pool2.shape) / p[2]
            self.pool2*=self.pool2_dropout
 
      self.CNN_layer3=self.ReLU(self.conv2d(self.pool2,self.layer3_filters)+self.CNN_layer3_bias)

      self.pool3=self.pooling((3,3),self.CNN_layer3,2)
      
      if training:
            self.pool3_dropout=cp.random.binomial(1, p[3], size=self.pool3.shape) / p[3]
            self.pool3*=self.pool3_dropout
      
      self.flatten=self.pool3.reshape(self.pool3.shape[0],self.pool3.shape[-1],1)
      
      self.MLP_layer=self.softmax(cp.matmul(self.MLP_weights,self.flatten)+self.MLP_bias)
      
      return cp.argmax(self.MLP_layer,axis=1)
    def findgradient(self,labels,training):
 
      labels=labels.reshape(labels.shape[0],10,1)
      curr_error=self.MLP_layer-labels
 
      
 
      self.MLP_weights_gradient=cp.matmul(curr_error,self.flatten.transpose(0,2,1))
      
      self.MLP_bias_gradient=curr_error
      
      curr_error=cp.matmul(self.MLP_weights.T,curr_error)
      curr_error=curr_error.reshape(curr_error.shape[0],self.pool3.shape[1],self.pool3.shape[2],curr_error.shape[-2])
      curr_error=curr_error*self.ReLUd(self.pool3)
      if training:
        curr_error*=self.pool3_dropout
      convert=curr_error.reshape(curr_error.shape[0],curr_error.shape[1],1,curr_error.shape[2],1,curr_error.shape[3])
      
      convert=(cp.tile(convert,(1,1,3,1,3,1))).reshape(self.CNN_layer3.shape[0],self.CNN_layer3.shape[1],self.CNN_layer3.shape[2],self.CNN_layer3.shape[3])
      curr_error=convert*self.inverse_pool_map[2]
      
      
      self.layer3_filters_gradient=self.conv2d_grad(self.pool2,curr_error)
      self.CNN_layer3_bias_gradient=cp.sum(curr_error,axis=(1,2))

      curr_error=self.full_conv(curr_error,self.layer3_filters)
      curr_error*=self.ReLUd(self.pool2)
      if training:
        curr_error*=self.pool2_dropout
      convert=curr_error.reshape(curr_error.shape[0],curr_error.shape[1],1,curr_error.shape[2],1,curr_error.shape[3])
     
      convert=(cp.tile(convert,(1,1,2,1,2,1))).reshape(self.CNN_layer2.shape[0],self.CNN_layer2.shape[1],self.CNN_layer2.shape[2],self.CNN_layer2.shape[3])
      curr_error=convert*self.inverse_pool_map[1]
      
      self.layer2_filters_gradient=self.conv2d_grad(self.pool1,curr_error)
      self.CNN_layer2_bias_gradient=cp.sum(curr_error,axis=(1,2))

      curr_error=self.full_conv(curr_error,self.layer2_filters)
      curr_error*=self.ReLUd(self.pool1)
      if training:
        curr_error*=self.pool1_dropout
      convert=curr_error.reshape(curr_error.shape[0],curr_error.shape[1],1,curr_error.shape[2],1,curr_error.shape[3])

      convert=(cp.tile(convert,(1,1,2,1,2,1))).reshape(self.CNN_layer1.shape[0],self.CNN_layer1.shape[1],self.CNN_layer1.shape[2],self.CNN_layer1.shape[3])
      curr_error=convert*self.inverse_pool_map[0]

      self.layer1_filters_gradient=self.conv2d_grad(self.input,curr_error)
      self.CNN_layer1_bias_gradient=cp.sum(curr_error,axis=(1,2))

    
      
      return 
    
    def gradientdescent(self,learning_rate,batch_size):
      
      self.MLP_weights-=learning_rate*cp.sum(self.MLP_weights_gradient,axis=0)*(1/batch_size)

      self.MLP_bias-=learning_rate*cp.sum(self.MLP_bias_gradient,axis=0)*(1/batch_size)

      self.layer3_filters-=learning_rate*self.layer3_filters_gradient*1/(batch_size)
      self.CNN_layer3_bias-=learning_rate*cp.sum(self.CNN_layer3_bias_gradient,axis=0)*1/(batch_size)
      self.layer2_filters-=learning_rate*self.layer2_filters_gradient*1/(batch_size)
      self.CNN_layer2_bias-=learning_rate*cp.sum(self.CNN_layer2_bias_gradient,axis=0)*1/(batch_size)

      self.layer1_filters-=learning_rate*self.layer1_filters_gradient*1/(batch_size)
      self.CNN_layer1_bias-=learning_rate*cp.sum(self.CNN_layer1_bias_gradient,axis=0)*1/(batch_size)
      
      return
    
    def train(self,images,labels,num_images,batch_size):
      learning_rate=0.04*batch_size/200
      s=time.time()
      self.run(images,[1,1,1,0.5],True)
      self.findgradient(labels,True)
      self.gradientdescent(learning_rate,batch_size)   
    
    def clear(self):
        self.CNN_layer1=self.CNN_layer2=self.CNN_layer3=self.pool1=self.pool2=self.pool3=self.pool1_dropout=self.pool2_dropout=self.pool3_dropout=self.flatten=self.MLP_layer=self.CNN_layer1_bias_gradient=self.layer1_filters_gradient=self.CNN_layer2_bias_gradient=self.layer2_filters_gradient=self.CNN_layer3_bias_gradient=self.layer3_filters_gradient=self.MLP_bias_gradient=self.MLP_weights_gradient=None   
 
    def checkaccuracy(self,x_test,y_test,num_images,typeloss):
       
        
        tot=x_test[0].shape[0]
        wrong=0
       
        iter=len(x_test)
        loss=0
        for i in range(iter):
            guess=self.run(cp.array(x_test[i]),0,False)
            guess=guess.reshape(guess.shape[0],)
            # self.loss_test+=((np.sum((self.layer1_weights)**2)+np.sum((self.layer2_weights)**2)+np.sum((self.layer3_weights)**2)) *(s/(2*num_images)) )
            ans=cp.array(y_test[i])
 
            layer_MLP=self.MLP_layer.reshape(self.MLP_layer.shape[0],self.MLP_layer.shape[1])
            loss+=cp.sum((layer_MLP-ans)**2) / num_images

            
               
            n_values=10
            
            one_hot_encoded_guess= cp.eye(n_values)[guess]
           
            one_hot_encoded_guess[cp.arange(guess.size),guess] = 1

            one_hot_encoded_guess=one_hot_encoded_guess.reshape(one_hot_encoded_guess.shape[0],n_values)
            
            wrong+=cp.sum(cp.max(ans-one_hot_encoded_guess,axis=1))
      
        print(typeloss + str(loss))
        return (1-wrong/num_images)*100
    
        
    
 

myNetwork=CNN([1,100,200,300,10],[(5,5),(3,3),(3,3)])
 
 
epoch=400
batch_size=32
num_images=60000
num_test_images=10000

num_batch_iteratons=num_images//batch_size 
 
(x_train,y),(x_test,y2)=tf.keras.datasets.fashion_mnist.load_data()
 
x_train=x_train.reshape(60000,28,28,1) 
x_test=x_test.reshape(10000,28,28,1)
x_train=x_train/255
x_test=x_test/255
x_train=np.array_split(x_train,60000//batch_size)
y_train = np.zeros((y.size, y.max()+1))
y_train[np.arange(y.size),y] = 1
y_train=np.array_split(y_train,60000//batch_size)
y_test = np.zeros((y2.size, y2.max()+1))
y_test[np.arange(y2.size),y2] = 1
x_test=np.array_split(x_test,10000//100)
 
y_test=np.array_split(y_test,10000//100)
 
begin_accuracy=myNetwork.checkaccuracy(x_test,y_test,num_test_images,"Initial test loss: ")
best_acc=begin_accuracy
print(begin_accuracy)
print("\n")
for x in range(epoch):
  
    s=time.time()
    
    for i in range(num_batch_iteratons):
        
        
        myNetwork.train(cp.array(x_train[i]),cp.array(y_train[i]),num_images,batch_size)
   
    print('Epoch '+ str(x+1)+' Done | Time Taken: '+ str(time.time()-s)+"s")
    cp._default_memory_pool.free_all_blocks()
    train_acc=myNetwork.checkaccuracy(x_train,y_train,num_images,"Train loss: ")
    test_acc=myNetwork.checkaccuracy(x_test,y_test,num_test_images,"Test loss: ")
    if test_acc>best_acc:
      print("New Best: " + str(test_acc))
      
      best_acc=test_acc
    print("\n")
