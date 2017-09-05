import tensorflow as tf
import numpy as np
import os
import glob
from joblib import Parallel, delayed
from tqdm import tqdm

base_folder = '/mnt/fs1/datasets/eight_world_dataset/new_tfdata/'
input_data = 'kNN'
input_dtype = np.uint16
output_data = 'kNN2'
output_dtype = np.int16

def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_record(record, base_folder, input_data, 
        input_dtype, output_data, output_dtype):
    input_datum = tf.train.Example()
    reader = tf.python_io.tf_record_iterator(path=
            os.path.join(base_folder, input_data, record))
    writer = tf.python_io.TFRecordWriter(
            os.path.join(base_folder, output_data, record))
    while(True):
        try:
            data = reader.next()
        except StopIteration:
            writer.close()
            break
        input_datum.ParseFromString(data)
        data = input_datum.features.feature[input_data].bytes_list.value[0]
        data = np.fromstring(data, dtype=input_dtype)
        data = data.astype(output_dtype)
        output_datum = tf.train.Example(features = tf.train.Features(
            feature = {input_data: _bytes_feature(data.tostring())}))
        writer.write(output_datum.SerializeToString())

# input records
records = [record for record in os.listdir(os.path.join(base_folder, input_data)) \
        if record.endswith('.tfrecords')]
# output path
if not os.path.exists(os.path.join(base_folder, output_data)):
    os.mkdir(os.path.join(base_folder, output_data))
# read in, convert to target dtype and write new tfrecords
Parallel(n_jobs=32)(delayed(convert_record)(
        record=record,
        base_folder=base_folder,
        input_data=input_data,
        input_dtype=input_dtype, 
        ouput_data=output_data, 
        output_dtype=output_dtype) \
                for num, record in \
                enumerate(tqdm(records, desc='tfrecords')))
