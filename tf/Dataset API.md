# Dataset API

[原文地址](https://zhuanlan.zhihu.com/p/30751039)

在 TF 中读取数据一般有三种方法：
> 1. 使用placeholder读内存中的数据
> 2. 使用queue读硬盘中的数据
> 3. 使用 Dataset API

这里我们着重看Dataset。

#### 基本概念：Dataset与Iterator
###### Dataset API中的类图:
![Dataset API中的类图](https://pic2.zhimg.com/80/v2-f9f42cc5c00573f7baaa815795f1ce45_720w.jpg)

初学时，我们只需要关注两个最重要的基础类：Dataset和Iterator。
###### Dataset
Dataset可以看作是相同类型“元素”的有序列表。单个“元素”可以是向量，也可以是字符串、图片，甚至是tuple或者dict。
先以最简单的，Dataset的每一个元素是一个数字为例：
```python
import tensorflow as tf
import numpy as np

dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
```
###### Iterator
如何将这个dataset中的元素取出呢？方法是从Dataset中示例化一个Iterator，然后对Iterator进行迭代。  
> - 在非Eager模式下，读取上述dataset中元素的方法为：
```python
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(one_element))
```
> - 在Eager模式下，是通过tfe.Iterator(dataset)直接创建Iterator并迭代。迭代时可以直接取出值，不需要使用sess.run()：
```python
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

for one_element in tfe.Iterator(dataset):
    print(one_element)
```

#### 从内存中创建更复杂的Dataset
之前我们用tf.data.Dataset.from_tensor_slices创建了一个最简单的Dataset：
```python
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
```
其实，tf.data.Dataset.from_tensor_slices的功能不止如此，它的真正作用是切分传入Tensor的第一个维度，生成相应的dataset。

例如：
Dataset中的每个元素是一个Python中的元组，或是Python中的词典。例如，在图像识别问题中，一个元素可以是{"image": image_tensor, "label": label_tensor} 的形式，这样处理起来更方便。

tf.data.Dataset.from_tensor_slices同样支持创建这种dataset，例如我们可以让每一个元素是一个词典：
```python
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    }
)
```
这时函数会分别切分"a"中的数值以及"b"中的数值，最终dataset中的一个元素就是类似于{"a": 1.0, "b": [0.9, 0.1]}的形式。

#### 对Dataset中的元素做变换：Transformation
常用的Transformation有：
> - map
> - batch
> - shuffle
> - repeat

###### （1）map
map接收一个函数，Dataset中的每个元素都会被当作这个函数的输入，并将函数返回值作为新的Dataset，如我们可以对dataset中每个元素的值加1：
```python
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.map(lambda x: x + 1) # 2.0, 3.0, 4.0, 5.0, 6.0
```
###### （2）batch

batch就是将多个元素组合成batch，如下面的程序将dataset中的每个元素组成了大小为32的batch：

```python
dataset = dataset.batch(32)
```

###### （3）shuffle

shuffle的功能为打乱dataset中的元素，它有一个参数buffersize，表示打乱时使用的buffer的大小：

```python
dataset = dataset.shuffle(buffer_size=10000)
```

###### （4）repeat

repeat的功能就是将整个序列重复多次，主要用来处理机器学习中的epoch，假设原先的数据是一个epoch，使用repeat(5)就可以将之变成5个epoch：

```python
dataset = dataset.repeat(5)
```
如果直接调用repeat()的话，生成的序列就会无限重复下去，没有结束，因此也不会抛出tf.errors.OutOfRangeError异常。

#### Dataset的其它创建方法...
除了tf.data.Dataset.from_tensor_slices外，目前Dataset API还提供了另外三种创建Dataset的方式：
> - tf.data.TextLineDataset()：这个函数的输入是一个文件的列表，输出是一个dataset。dataset中的每一个元素就对应了文件中的一行。可以使用这个函数来读入CSV文件。
> - tf.data.FixedLengthRecordDataset()：这个函数的输入是一个文件的列表和一个record_bytes，之后dataset的每一个元素就是文件中固定字节数record_bytes的内容。通常用来读取以二进制形式保存的文件，如CIFAR10数据集就是这种形式。
> - tf.data.TFRecordDataset()：顾名思义，这个函数是用来读TFRecord文件的，dataset中的每一个元素就是一个TFExample。

#### 更多类型的Iterator...
在非Eager模式下，最简单的创建Iterator的方法就是通过dataset.make_one_shot_iterator()来创建一个one shot iterator。除了这种one shot iterator外，还有三个更复杂的Iterator，即：
> - initializable iterator
>> initializable iterator必须要在使用前通过sess.run()来初始化。使用initializable iterator，可以将placeholder代入Iterator中，这可以方便我们通过参数快速定义新的Iterator。一个简单的initializable iterator使用示例：
```python
 limit = tf.placeholder(dtype=tf.int32, shape=[])

dataset = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=limit))

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={limit: 10})
    for i in range(10):
      value = sess.run(next_element)
      assert i == value
```
> - reinitializable iterator
> - feedable iterator
