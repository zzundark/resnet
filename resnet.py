import tensorflow as tf
import config as cfg
import numpy as np
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope


#Batch Norm
def Batch_Norm(x, training, scope="bn"):
	with arg_scope([batch_norm],
					scope=scope,
					updates_collections=None,
					decay=0.9,
					center=True,
					scale=True,
					zero_debias_moving_mean=True):
		return tf.cond(training,
						lambda : batch_norm(inputs=x, is_training=training, reuse=None),
						lambda : batch_norm(inputs=x, is_training=training, reuse=True))
#Global_Average_Pooling
def Global_Average_Pooling(x, stride=1):
    
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) 
	
#모델(model)
class resnet(object):	
		
	def __init__(self):
		self.num_class = cfg.num_class
		self.training=tf.placeholder(tf.bool)
		self.images = tf.placeholder(tf.float32, shape=[None,cfg.image_size,cfg.image_size,3], name = 'x_image') 
		self.labels = tf.placeholder(tf.float32, shape=[None, self.num_class], name = 'y_target')
		self.logits = self.build_network(x_image=self.images,num_class=self.num_class)
		
	def conv_layer(self,x, input_dim,kernel_size,stride,output_size,scope):
		W = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, input_dim, output_size], stddev=0.1))
		b = tf.Variable(tf.zeros([output_size]),name="bias_"+scope)
		_x = Batch_Norm(x,self.training, scope="bn_"+scope)
		_x = tf.nn.relu(_x)
		_x = tf.nn.conv2d(_x, W, strides=[1, stride, stride, 1], padding='SAME') + b
		return _x
	def linear_layer(self,x, input_dim,output_size,scope):
		W = tf.Variable(tf.truncated_normal([input_dim, output_size], stddev=0.1))
		b = tf.Variable(tf.zeros([output_size]),name="bias_"+scope)
		_x = tf.matmul(x, W) + b
		return _x			
	def cnn(self,x_data,scope):
		divide=cfg.divide
		x=x_data
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,7,2,int(64/divide),scope="conv1_"+scope)
		print(x.shape)
		
		x=tf.nn.max_pool(x , ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
		skip_x=x
		
		
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(64/divide),scope="conv2_1"+scope)
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(64/divide),scope="conv2_2"+scope)+skip_x
		skip_x=x

		
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(64/divide),scope="conv2_3"+scope)
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(64/divide),scope="conv2_4"+scope)+skip_x
		skip_x=x
		skip_x = tf.nn.avg_pool(skip_x, ksize=[1, 2, 2, 1],
								strides=[1, 2, 2, 1], padding='VALID') 
		skip_x = tf.pad(skip_x, [[0, 0], [0, 0], [0, 0], [input_num // 2,
															input_num // 2]])
																	 
		print(x.shape)
		
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,2,int(128/divide),scope="conv3_1"+scope)	
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(128/divide),scope="conv3_2"+scope)+skip_x
		skip_x=x				
		
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(128/divide),scope="conv3_3"+scope)
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(128/divide),scope="conv3_4"+scope)+skip_x
		skip_x=x
		skip_x = tf.nn.avg_pool(skip_x, ksize=[1, 2, 2, 1],
								strides=[1, 2, 2, 1], padding='VALID') 
		skip_x = tf.pad(skip_x, [[0, 0], [0, 0], [0, 0], [input_num // 2,
															input_num // 2]])	
		
		print(x.shape)
		
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,2,int(256/divide),scope="conv4_1"+scope)
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(256/divide),scope="conv4_2"+scope)+skip_x
																
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(256/divide),scope="conv4_3"+scope)
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(256/divide),scope="conv4_4"+scope)+skip_x
		skip_x=x
		skip_x = tf.nn.avg_pool(skip_x, ksize=[1, 2, 2, 1],
								strides=[1, 2, 2, 1], padding='VALID') 
		skip_x = tf.pad(skip_x, [[0, 0], [0, 0], [0, 0], [input_num // 2,
															input_num // 2]])		
		print(x.shape)
		
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,2,int(512/divide),scope="conv5_1"+scope)
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(512/divide),scope="conv5_2"+scope)+skip_x	
		skip_x=x
		
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(512/divide),scope="conv5_3"+scope)
		x_shape = x.get_shape().as_list()
		input_num = x_shape[3]
		x=self.conv_layer(x,input_num,3,1,int(512/divide),scope="conv5_4"+scope)+skip_x	
		print(x.shape)
		
		x=Global_Average_Pooling(x)
		print(x.shape)

		x_shape = x.get_shape().as_list()
		x = tf.reshape(x,[-1, x_shape[1] * x_shape[2] * x_shape[3]])
		return x
		

		
	def build_network(self,x_image,num_class):
	
		x=self.cnn(x_image,"resnet")
		x_shape = x.get_shape().as_list()
		input_num= x_shape[1] 
		x=self.linear_layer(x,input_num,num_class,scope="out")
		return x		
		
		
		
