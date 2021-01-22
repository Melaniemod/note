# 从 Keras 到 Estimator

[原文链接](https://developer.aliyun.com/article/704422)

背景：使用TF跑Libsvm格式数据集

面临三个选择:
> 1. Tensorflow API：  
用低阶 API 也能马上建个 DNN，只是代码略显杂乱，相比之下，高阶 API Keras 代码极度精简，一目了然。  
> 2. Libsvm 数据读取：  
手写个 Libsvm 格式数据的读取很容易，读取稀疏编码转成稠密编码，但既然 sklearn 已经有 load_svmlight_file 了为什么不用呢，该函数会读进整个文件，当然小数据量不是问题。  
> 3. fit 和 fit_generator：  
Keras 模型训练只接收稠密编码，而 Libsvm 是稀疏编码，如果数据集不算太大，通过 load_svmlight_file 全部读进内存也能接受，但要先全部转成稠密编码再喂给 fit，那内存可能会爆掉；  
理想方案是用多少读多少，读进来再转换，此处图省事就先用 load_svmlight_file 全部读进来以稀疏编码保存，使用时再分批喂给 fit_generator。


```python
import numpy as np
from sklearn.datasets import load_svmlight_file
from tensorflow import keras
import tensorflow as tf

feature_len = 100000 # 特征维度，下面使用时可替换成 X_train.shape[1]
n_epochs = 1
batch_size = 256
train_file_path = './data/train_libsvm.txt'
test_file_path = './data/test_libsvm.txt'

def batch_generator(X_data, y_data, batch_size):
    number_of_batches = X_data.shape[0]/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while True:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        y_batch = y_data[index_batch]
        counter += 1
        yield np.array(X_batch),y_batch
        if (counter > number_of_batches):
            counter=0

def create_keras_model(feature_len):
    model = keras.Sequential([
        # 可在此添加隐层
        keras.layers.Dense(64, input_shape=[feature_len], activation=tf.nn.tanh),
        keras.layers.Dense(6, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X_train, y_train = load_svmlight_file(train_file_path)
    X_test, y_test = load_svmlight_file(test_file_path)

    keras_model = create_keras_model(X_train.shape[1])

    keras_model.fit_generator(generator=batch_generator(X_train, y_train, batch_size = batch_size),
                    steps_per_epoch=int(X_train.shape[0]/batch_size),
                    epochs=n_epochs)
    
    test_loss, test_acc = keras_model.evaluate_generator(generator=batch_generator(X_test, y_test, batch_size = batch_size),
                    steps=int(X_test.shape[0]/batch_size))
    print('Test accuracy:', test_acc)
```
上面代码的缺点显而易见：
> 1. 空间复杂度太差，大数据常驻内存会影响其它进程，当遇到大数据集时就无能为力了；
> 2. 可用性差，数据分批需在 batch_generator 手动编码实现，调试耗费时间，也容易出错。


#### 首先附上 embedding 代码，第一个隐层的输出作为 embedding

```python

def save_output_file(output_array, filename):
    result = list()
    for row_data in output_array:
        line = ','.join([str(x) for x in row_data.tolist()])
        result.append(line)
    with open(filename,'w') as fw:
        fw.write('%s' % '\n'.join(result))
        
X_test, y_test = load_svmlight_file("./data/test_libsvm.txt")
model = load_model('./dnn_onelayer_tanh.model')
dense1_layer_model = Model(inputs=model.input, outputs=model.layers[0].output)
dense1_output = dense1_layer_model.predict(X_test)
save_output_file(dense1_output, './hidden_output/hidden_output_test.txt')
```


#### Dataset 应用在 Keras model
```python
import numpy as np
from sklearn.datasets import load_svmlight_file
from tensorflow import keras
import tensorflow as tf

feature_len = 138830
n_epochs = 1
batch_size = 256
train_file_path = './data/train_libsvm.txt'
test_file_path = './data/test_libsvm.txt'

def decode_libsvm(line):
    columns = tf.string_split([line], ' ')
    labels = tf.string_to_number(columns.values[0], out_type=tf.int32)
    labels = tf.reshape(labels,[-1])
    splits = tf.string_split(columns.values[1:], ':')
    id_vals = tf.reshape(splits.values,splits.dense_shape)
    feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
    feat_ids = tf.string_to_number(feat_ids, out_type=tf.int64)
    feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
    # 由于 libsvm 特征编码从 1 开始，这里需要将 feat_ids 减 1
    sparse_feature = tf.SparseTensor(feat_ids-1, tf.reshape(feat_vals,[-1]), [feature_len])
    dense_feature = tf.sparse.to_dense(sparse_feature)
    return dense_feature, labels

def create_keras_model():
    model = keras.Sequential([
        keras.layers.Dense(64, input_shape=[feature_len], activation=tf.nn.tanh),
        keras.layers.Dense(6, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

if __name__ == "__main__":
    dataset_train = tf.data.TextLineDataset([train_file_path]).map(decode_libsvm).batch(batch_size).repeat()
    dataset_test = tf.data.TextLineDataset([test_file_path]).map(decode_libsvm).batch(batch_size).repeat()

    keras_model = create_keras_model()

    sample_size = 10000 # 由于训练函数必须要指定 steps_per_epoch，所以这里需要先获取到样本数
    keras_model.fit(dataset_train, steps_per_epoch=int(sample_size/batch_size), epochs=n_epochs)
    
    test_loss, test_acc = keras_model.evaluate(dataset_test, steps=int(sample_size/batch_size))
    print('Test accuracy:', test_acc)

```
这种方式解决了空间复杂度高的问题，不占用大量内存。
不过可用性上仍有两点不便：
> 1. keras fit 时需指定 steps_per_epoch，为了保证每一轮走完整批数据。
> 2. 需事先计算特征维度 feature_len，由于 libsvm 是稀疏编码，只读取一行或几行无法推断特征维度，可先离线用 load_svmlight_file 获取特征维度，然后写死在代码里。这是 libsvm 的固有特点，只能如此处理了。


#### Dataset 应用在 Keras to estimator
Estimator 是跟 Keras 相互独立的高阶 API，如果之前用的是 Keras，一时半会不能全部重构成 Estimator， TF 还提供了 Keras 的 model_to_estimator 接口，也可以享受到 Estimator 带来的好处。
```python
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.platform import tf_logging
# 打开 estimator 日志，可在训练时输出日志，了解进度
tf_logging.set_verbosity('INFO')

feature_len = 100000
n_epochs = 1
batch_size = 256
train_file_path = './data/train_libsvm.txt'
test_file_path = './data/test_libsvm.txt'

# 注意这里多了个参数 input_name，返回值也与上不同
# 这里 input_name 是干嘛来着？
def decode_libsvm(line, input_name):
    columns = tf.string_split([line], ' ')
    labels = tf.string_to_number(columns.values[0], out_type=tf.int32)
    labels = tf.reshape(labels,[-1])
    splits = tf.string_split(columns.values[1:], ':')
    id_vals = tf.reshape(splits.values,splits.dense_shape)
    feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
    feat_ids = tf.string_to_number(feat_ids, out_type=tf.int64)
    feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
    sparse_feature = tf.SparseTensor(feat_ids-1, tf.reshape(feat_vals,[-1]),[feature_len])
    dense_feature = tf.sparse.to_dense(sparse_feature)
    return {input_name: dense_feature}, labels

def input_train(input_name):
    # 这里使用 lambda 来给 map 中的 decode_libsvm 函数添加除 line 之的参数
    return tf.data.TextLineDataset([train_file_path]).map(lambda line: decode_libsvm(line, input_name)).batch(batch_size).repeat(n_epochs).make_one_shot_iterator().get_next()

def input_test(input_name):
    return tf.data.TextLineDataset([train_file_path]).map(lambda line: decode_libsvm(line, input_name)).batch(batch_size).make_one_shot_iterator().get_next()

def create_keras_model(feature_len):
    model = keras.Sequential([
        # 可在此添加隐层
        keras.layers.Dense(64, input_shape=[feature_len], activation=tf.nn.tanh),
        keras.layers.Dense(6, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

def create_keras_estimator():
    model = create_keras_model(feature_len)
    input_name = model.input_names[0]
    estimator = tf.keras.estimator.model_to_estimator(model)
    return estimator, input_name

if __name__ == "__main__":
    keras_estimator, input_name = create_keras_estimator(feature_len)
    keras_estimator.train(input_fn=lambda:input_train(input_name))
    eval_result = keras_estimator.evaluate(input_fn=lambda:input_train(input_name))
    print(eval_result)
```
注意到 Estimator 的 input_fn 返回的 dict key 需要跟 model 的输入名保持一致，这里通过 input_name 传递该值。




#### Dataset 应用在 DNNClassifier
直接使用 Tensorflow 预创建的 Estimator：DNNClassifier。
```python
import tensorflow as tf
from tensorflow.python.platform import tf_logging
# 打开 estimator 日志，可在训练时输出日志，了解进度
tf_logging.set_verbosity('INFO')

feature_len = 100000
n_epochs = 1
batch_size = 256
train_file_path = './data/train_libsvm.txt'
test_file_path = './data/test_libsvm.txt'

def decode_libsvm(line, input_name):
    columns = tf.string_split([line], ' ')
    labels = tf.string_to_number(columns.values[0], out_type=tf.int32)
    labels = tf.reshape(labels,[-1])
    splits = tf.string_split(columns.values[1:], ':')
    id_vals = tf.reshape(splits.values,splits.dense_shape)
    feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
    feat_ids = tf.string_to_number(feat_ids, out_type=tf.int64)
    feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
    sparse_feature = tf.SparseTensor(feat_ids-1,tf.reshape(feat_vals,[-1]),[feature_len])
    dense_feature = tf.sparse.to_dense(sparse_feature)
    return {input_name: dense_feature}, labels

def input_train(input_name):
    return tf.data.TextLineDataset([train_file_path]).map(lambda line: decode_libsvm(line, input_name)).batch(batch_size).repeat(n_epochs).make_one_shot_iterator().get_next()

def input_test(input_name):
    return tf.data.TextLineDataset([train_file_path]).map(lambda line: decode_libsvm(line, input_name)).batch(batch_size).make_one_shot_iterator().get_next()

def create_dnn_estimator():
    input_name = "dense_input"
    feature_columns = tf.feature_column.numeric_column(input_name, shape=[feature_len])
    estimator = tf.estimator.DNNClassifier(hidden_units=[64],
                                           n_classes=6,
                                           feature_columns=[feature_columns])
    return estimator, input_name

if __name__ == "__main__":
    dnn_estimator, input_name = create_dnn_estimator()
    dnn_estimator.train(input_fn=lambda:input_train(input_name))

    eval_result = dnn_estimator.evaluate(input_fn=lambda:input_test(input_name))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
```