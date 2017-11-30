from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


# Define Sliding Window
sw_length = 60 # minutes
sw_step = 5 # minute

sw_start_time = dt.datetime.strptime('2017-11-27 01:00:00','%Y-%m-%d %H:%M:%S')
sw_end_time = sw_start_time + dt.timedelta(minutes=sw_length)

max_rate = 2
min_rate = 0.01

def normalizeRate(rate):
    return (rate-min_rate)/(max_rate-min_rate)


# Training Data
print('Prepare training data...')
input_data = []
normal_log = pd.read_csv("./apigw/normal_log.csv", header=None)
normal_log.columns = ['id', 'type', 'datetime']
normal_log['datetime'] = pd.to_datetime(normal_log['datetime']) #Set data type to datetime
normal_log = normal_log.set_index('datetime') # Set date as index

for i in range(2900):
    sw_df = normal_log[sw_start_time : sw_end_time]
    #print('start:{0}, end:{1}'.format(sw_start_time, sw_end_time))
    
    traffic_counts = sw_df['type'].count()
    traffic_type_counts = sw_df['type'].value_counts()

    # Calculate feature: mo_traffic_percentage
    if 'NIDD-MO' in traffic_type_counts:
        mo_traffic_percentage = traffic_type_counts['NIDD-MO'] / traffic_counts
    else:
        mo_traffic_percentage = 0.0
    #print('mo_traffic_percentage:{0}'.format(mo_traffic_percentage))

    # Calculate feature: mt_traffic_percentage
    if 'NIDD-MT' in traffic_type_counts:
        mt_traffic_percentage = sw_df['type'].value_counts()['NIDD-MT']/ traffic_counts
    else:
        mt_traffic_percentage = 0.0
    #print('mt_traffic_percentage:{0}'.format(mt_traffic_percentage))
    
    # Calculate feature: err_traffic_percentage
    if 'ERROR' in traffic_type_counts:
        err_traffic_percentage = sw_df['type'].value_counts()['ERROR']/ traffic_counts
    else:
        err_traffic_percentage = 0.0
    #print('err_traffic_percentage:{0}'.format(err_traffic_percentage))


    max_message_counts = sw_df['id'].value_counts().max()
    max_mo_rate_per_device = normalizeRate(max_message_counts / 60)

    #print('max_mo_rate_per_device:{0}'.format(max_mo_rate_per_device))

    input_data.append([mo_traffic_percentage, mt_traffic_percentage, err_traffic_percentage, max_mo_rate_per_device])

    sw_start_time += dt.timedelta(minutes=sw_step)
    sw_end_time += dt.timedelta(minutes=sw_step)

#print(input_data)
print('Training data preparation done!')

# Testing Data
print('Prepare testing data...')
abnormal_log = pd.read_csv("./apigw/abnormal_log.csv", header=None)
abnormal_log.columns = ['id', 'type', 'datetime']
abnormal_log['datetime'] = pd.to_datetime(abnormal_log['datetime']) #将数据类型转换为日期类型
abnormal_log = abnormal_log.set_index('datetime') # 将date设置为index

test_data = []
sw_start_time = dt.datetime.strptime('2017-12-20 01:00:00','%Y-%m-%d %H:%M:%S')
sw_end_time = sw_start_time + dt.timedelta(minutes=sw_length)

for i in range(2000):
    sw_df = abnormal_log[sw_start_time : sw_end_time]
    #print('start:{0}, end:{1}'.format(sw_start_time, sw_end_time))
    
    traffic_counts = sw_df['type'].count()
    traffic_type_counts = sw_df['type'].value_counts()

    # Calculate feature: mo_traffic_percentage
    if 'NIDD-MO' in traffic_type_counts:
        mo_traffic_percentage = traffic_type_counts['NIDD-MO'] / traffic_counts
    else:
        mo_traffic_percentage = 0.0
    #print('mo_traffic_percentage:{0}'.format(mo_traffic_percentage))

    # Calculate feature: mt_traffic_percentage
    if 'NIDD-MT' in traffic_type_counts:
        mt_traffic_percentage = sw_df['type'].value_counts()['NIDD-MT']/ traffic_counts
    else:
        mt_traffic_percentage = 0.0
    #print('mt_traffic_percentage:{0}'.format(mt_traffic_percentage))
    
    # Calculate feature: err_traffic_percentage
    if 'ERROR' in traffic_type_counts:
        err_traffic_percentage = sw_df['type'].value_counts()['ERROR']/ traffic_counts
    else:
        err_traffic_percentage = 0.0
    #print('err_traffic_percentage:{0}'.format(err_traffic_percentage))


    max_message_counts = sw_df['id'].value_counts().max()
    max_mo_rate_per_device = normalizeRate(max_message_counts / 60)

    #print('max_mo_rate_per_device:{0}'.format(max_mo_rate_per_device))

    test_data.append([mo_traffic_percentage, mt_traffic_percentage, err_traffic_percentage, max_mo_rate_per_device])

    sw_start_time += dt.timedelta(minutes=sw_step)
    sw_end_time += dt.timedelta(minutes=sw_step)

test_data_array = np.array(test_data)
label = ["Max Rate per Device", "MO Traffic", "MT Traffic", "Error Traffic", "Output"]  
plt.plot(test_data_array[:, 3], "b")
plt.plot(test_data_array[:, 0], "green")
plt.plot(test_data_array[:, 1], "yellow")
plt.plot(test_data_array[:, 2], "black")
#print(test_data)

print('Testing data preparation done!')

# Training Parameters
learning_rate = 0.01
num_steps = 290
batch_size = 10

display_step = 10

# Network Parameters
num_hidden_1 = 4 # 1st layer num features
num_hidden_2 = 4 # 2nd layer num features
num_input = 4

# tf Graph input
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

out_put_data = []

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    with open('C://Ericsson//Projects//tensorflow//apigw//log.txt', 'wt') as f:
        # Training
        print('Start traning...')
        for i in range(1, num_steps+1):
            # Prepare Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_index_start = (i-1)*batch_size
            batch_index_end = i*batch_size-1
            batch_x = input_data[batch_index_start : batch_index_end]
            #print('{0} : {1}'.format(batch_index_start, batch_index_end))
            #print('batch_x:{0}'.format(batch_x))

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l), file=f)
        print('Training Done!')

        print('Start testing...')
        # Testing
        for i in range(0, 2000):
            #batch_index_start = (i-1)*batch_size
            #batch_index_end = i*batch_size-1
            batch_x = test_data[i : i+1]
            print('batch_x:{0}'.format(batch_x), file=f)
            # Encode and decode
            g = sess.run(decoder_op, feed_dict={X: batch_x})
            mse = np.mean((g-batch_x)**2)
            out_put_data.append(mse)
            print('mse:{0}'.format(mse), file=f)
        print('Training Done!')

  # Export model
    export_path = "C://Ericsson//Projects//tensorflow//apigw//model"
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
    builder.save()
    print('Exporting Done!')

print('Plotting the graph...')
plt.plot(out_put_data, "r")
plt.legend(label, loc = 0, ncol = 2)
plt.show()
