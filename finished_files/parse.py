import struct
import sys
import pickle
import tensorflow as tf
from tensorflow.core.example import example_pb2
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def _binary_to_text():
    reader = open('train.bin', 'rb')
    writer = open('train.txt', 'w')
    dataset = []
    while True:
        len_bytes = reader.read(8)
        if not len_bytes:
            sys.stderr.write('Done reading\n')
            with open('train.pickle', 'wb') as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('train.pickle', 'rb') as handle:
                b = pickle.load(handle)
                print dataset==b
            return
        str_len = struct.unpack('q', len_bytes)[0]
        tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        tf_example = example_pb2.Example.FromString(tf_example_str)
        examples = []
        data = {}
        for key in tf_example.features.feature:
            data[key] = tf_example.features.feature[key].bytes_list.value[0]
        dataset.append(data)
        #writer.write('%s\n' % '\t'.join(examples))
    
    print a == b
    reader.close()
    writer.close()
_binary_to_text()