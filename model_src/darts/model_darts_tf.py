from functools import partial
from model_src.darts.operations_tf import *


def Cell(s0,s1,genotype, C_out, reduction, reduction_prev):
	if reduction:
		op_names, indices = zip(*genotype.reduce)
		concat = genotype.reduce_concat
	else:
		op_names, indices = zip(*genotype.normal)
		concat = genotype.normal_concat

	cells_num=len(op_names) // 2
	multiplier = len(concat)

	if reduction_prev:
		s0 = FactorizedReduce(s0,C_out)
	else:
		s0 = ReLUConvBN(s0,C_out)
	s1=ReLUConvBN(s1,C_out)

	state=[s0,s1]
	offset=0
	for i in range(cells_num):
		temp=[]
		for j in range(2):
			stride = [2,2] if reduction and indices[2*i+j] < 2 else [1,1]
			h = state[indices[2*i+j]]
			temp.append(OPS[op_names[2*i+j]](h, C_out, stride))
			#did not implement path drop
		state.append(tf.add_n(temp))
	out=tf.concat(state[-multiplier:],axis=-1)
	return out


def AuxiliaryHeadCIFAR(x,class_num):
	x=tflearn.relu(x)
	x=slim.avg_pool2d(x,[5,5], stride=3,padding='SAME')
	x=slim.conv2d(x,128,[1,1])
	x=slim.batch_norm(x)
	x=tflearn.relu(x)
	x=slim.conv2d(x,768,[2,2])
	x=slim.batch_norm(x)
	x=tflearn.relu(x)
	x=slim.flatten(x)
	x=slim.fully_connected(x,class_num, activation_fn=None)
	return x


def Model(x,is_training,first_C,class_num,layer_num,auxiliary,genotype,stem_multiplier=3):
	with tf.variable_scope('lw',reuse=tf.AUTO_REUSE):
		with slim.arg_scope([slim.conv2d,slim.separable_conv2d], activation_fn=None,
                            padding='SAME', biases_initializer=None, weights_regularizer=slim.l2_regularizer(0.0001)):
			with slim.arg_scope([slim.batch_norm],is_training=is_training):
				C_curr = stem_multiplier*first_C
				s0 =slim.conv2d(x,C_curr,[3,3],activation_fn=tflearn.relu)
				s0=slim.batch_norm(s0)
				s1 =slim.conv2d(x,C_curr,[3,3],activation_fn=tflearn.relu)
				s1=slim.batch_norm(s1)
				reduction_prev = False
				logits_aux=None
				for i in range(layer_num):
					if i in [layer_num//3, 2*layer_num//3]:
						C_curr *= 2
						reduction = True
					else:
						reduction = False
					s0, s1 = s1, Cell(s0, s1, genotype, C_curr, reduction, reduction_prev)
					reduction_prev = reduction
					if auxiliary and i == 2*layer_num//3:
						logits_aux=AuxiliaryHeadCIFAR(s1,class_num)

				try:
					out=tf.reduce_mean(s1, [1, 2], keep_dims=True, name='global_pool')
				except TypeError:
					out = tf.reduce_mean(s1, [1, 2], keepdims=True, name='global_pool')

				logits = slim.conv2d(out, class_num, [1, 1], activation_fn=None, normalizer_fn=None,
                                     weights_regularizer=slim.l2_regularizer(0.0001))
				logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
	return logits,logits_aux


def AuxiliaryHeadImageNet(x, class_num):
	x=tflearn.relu(x)
	x=slim.avg_pool2d(x,[5,5], stride=3,padding='SAME')
	x=slim.conv2d(x,128,[1,1])
	x=slim.batch_norm(x)
	x=tflearn.relu(x)
	x=slim.conv2d(x,768,[2,2])
	# x=slim.batch_norm(x) Author of DARTS code commented this out due to a typo
	x=tflearn.relu(x)
	x=slim.flatten(x)
	x=slim.fully_connected(x,class_num, activation_fn=None)
	return x


def ModelImageNet(x,is_training,first_C,class_num,layer_num,auxiliary,genotype,stem_multiplier=3):
	with tf.variable_scope('lw',reuse=tf.AUTO_REUSE):
		with slim.arg_scope([slim.conv2d,slim.separable_conv2d],activation_fn=None,padding='SAME',
							biases_initializer=None,weights_regularizer=slim.l2_regularizer(0.0001)):
			with slim.arg_scope([slim.batch_norm],is_training=is_training):
				C_curr = stem_multiplier*first_C

				# Stem 0
				s0 = slim.conv2d(x, num_outputs=C_curr // 2, kernel_size=3, stride=2)
				s0 = slim.batch_norm(s0)
				s0 = tflearn.relu(s0)
				s0 = slim.conv2d(s0, num_outputs=C_curr, kernel_size=3, stride=2)
				s0 = slim.batch_norm(s0)

				# Stem 1
				s1 = tflearn.relu(s0)
				s1 = slim.conv2d(s1, num_outputs=C_curr, kernel_size=3, stride=2)
				s1 = slim.batch_norm(s1)

				reduction_prev = True
				logits_aux=None
				for i in range(layer_num):
					if i in [layer_num//3, 2*layer_num//3]:
						C_curr *= 2
						reduction = True
					else:
						reduction = False
					s0,s1 =s1,Cell(s0,s1,genotype, C_curr, reduction, reduction_prev)
					reduction_prev = reduction
					if auxiliary and i == 2*layer_num//3:
						logits_aux=AuxiliaryHeadImageNet(s1,class_num)

				try:
					out = tf.reduce_mean(s1, [1, 2], keep_dims=True, name='global_pool')
				except TypeError:
					out = tf.reduce_mean(s1, [1, 2], keepdims=True, name='global_pool')

				logits = slim.conv2d(out, class_num, [1, 1], activation_fn=None,normalizer_fn=None,
									 weights_regularizer=slim.l2_regularizer(0.0001))
				logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
	return logits,logits_aux


def cifar10_model(x, genotype, is_training,
				  first_C=16, class_num=10,
				  layer_num=20, stem_multiplier=3): # No Aux head
	with slim.arg_scope([slim.conv2d, slim.separable_conv2d], activation_fn=None, padding='SAME',
						biases_initializer=None, weights_regularizer=slim.l2_regularizer(0.0001)):
		with slim.arg_scope([slim.batch_norm], is_training=is_training):
			x = tf.identity(x, "input")
			C_curr = stem_multiplier * first_C
			s0 = slim.conv2d(x, C_curr, [3, 3], activation_fn=tflearn.relu)
			s0 = slim.batch_norm(s0)
			s1 = slim.conv2d(x, C_curr, [3, 3], activation_fn=tflearn.relu)
			s1 = slim.batch_norm(s1)
			reduction_prev = False
			for i in range(layer_num):
				if i in [layer_num // 3, 2 * layer_num // 3]:
					C_curr *= 2
					reduction = True
				else:
					reduction = False
				s0, s1 = s1, Cell(s0, s1, genotype, C_curr, reduction, reduction_prev)
				reduction_prev = reduction

			try:
				out = tf.reduce_mean(s1, [1, 2], keep_dims=True, name='global_pool')
			except TypeError:
				out = tf.reduce_mean(s1, [1, 2], keepdims=True, name='global_pool')

			logits = slim.conv2d(out, class_num, [1, 1], activation_fn=None, normalizer_fn=None,
								 weights_regularizer=slim.l2_regularizer(0.0001))
			logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
	return logits


def cifar10_model_maker(genotype):
	forward = partial(cifar10_model, genotype=genotype)
	return lambda _x, training: forward(x=_x, is_training=training)
