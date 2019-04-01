import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
import argparse
from actor_critic import Actor


parser = argparse.ArgumentParser(description='Configuration file to train the model')
arg_lists = []

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')


################################################### DA MODIFICARE ###############################################
# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=128, help='batch size')
data_arg.add_argument('--graph_dimension', type=int, default=50, help='number of cities') ##### #####
data_arg.add_argument('--dimension', type=int, default=2, help='city dimension')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--input_embed', type=int, default=128, help='actor critic input embedding')
net_arg.add_argument('--num_neurons', type=int, default=512, help='encoder inner layer neurons')
net_arg.add_argument('--num_stacks', type=int, default=7, help='encoder num stacks')
net_arg.add_argument('--num_heads', type=int, default=16, help='encoder num heads')
net_arg.add_argument('--query_dim', type=int, default=360, help='decoder query space dimension')
net_arg.add_argument('--num_units', type=int, default=256, help='decoder and critic attention product space')
net_arg.add_argument('--num_neurons_critic', type=int, default=256, help='critic n-1 layer')

# Train / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--nb_steps', type=int, default=80000, help='nb steps')
train_arg.add_argument('--init_B', type=float, default=7., help='critic init baseline')
train_arg.add_argument('--lr_start', type=float, default=0.001, help='actor learning rate')
train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='lr1 decay step')
train_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='lr1 decay rate')
train_arg.add_argument('--temperature', type=float, default=1.0, help='pointer initial temperature')
train_arg.add_argument('--C', type=float, default=10.0, help='pointer tanh clipping')
train_arg.add_argument('--is_training', type=str2bool, default=True, help='switch to inference mode when model is trained') 

parser.parse_args()

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed



config, _ = get_config()
dir_ = str(config.dimension)+'D_'+'TSP'+str(config.graph_dimension) +'_b'+str(config.batch_size)+'_e'+str(config.input_embed)+'_n'+str(config.num_neurons)+'_s'+str(config.num_stacks)+'_h'+str(config.num_heads)+ '_q'+str(config.query_dim) +'_u'+str(config.num_units)+'_c'+str(config.num_neurons_critic)+ '_lr'+str(config.lr_start)+'_d'+str(config.lr_decay_step)+'_'+str(config.lr_decay_rate)+ '_T'+str(config.temperature)+ '_steps'+str(config.nb_steps)+'_i'+str(config.init_B) 
print(dir_)




                
                
tf.reset_default_graph()
actor = Actor(config) # Build graph

variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name] # Save & restore all the variables.
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)   


with tf.Session() as sess: # start session
    sess.run(tf.global_variables_initializer()) # Run initialize op
    variables_names = [v.name for v in tf.trainable_variables() if 'Adam' not in v.name]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        #print("Variable: ", k, "Shape: ", v.shape) # print all variables
        pass
    


    
    
    
########################################## TRAIN #################################


from dataGenerator import  DataGenerator

dataset = DataGenerator()


np.random.seed(123) # reproducibility
tf.set_random_seed(123)


with tf.Session() as sess: # start session
    sess.run(tf.global_variables_initializer()) # run initialize op
    writer = tf.summary.FileWriter('summary/'+dir_, sess.graph) # summary writer
    
    for i in tqdm(range(config.nb_steps)): # Forward pass & train step
        input_batch, dist_batch = dataset.train_batch(actor.batch_size, actor.max_length, actor.dimension)
        feed = {actor.input_: input_batch, actor.distances: dist_batch} # get feed dict
        reward, predictions, summary, _, _ = sess.run([actor.reward, actor.predictions, 
                                                       actor.merged, actor.trn_op1, actor.trn_op2], feed_dict=feed)

        if i % 200 == 0: 
            print('reward',np.mean(reward))
            print('predictions',np.mean(predictions))
            writer.add_summary(summary,i)
        
    save_path = "save/"+dir_
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saver.save(sess, save_path+"/actor.ckpt") # save the variables to disk
    print("Training COMPLETED! Model saved in file: %s" % save_path)