# Estimator

[原文地址](https://zhuanlan.zhihu.com/p/41473323)

#### 概述
`tf.estimator.train_and_evaluate`简化了训练、评估和导出Estimator模型的过程，抽象了模型分布式训练和评估的细节，使得同样的代码在本地与分布式集群上的行为一致。  

这意味着使用train_and_evaluate API，我们可以在本地和分布式集群上、不同的设备和硬件上跑同样的代码，而不需要修改代码已适应不同的部署环境。而且训练之后的模型可以很方便地导出以便在打分服务（tensorflow serving）中使用。

本文简要介绍如何自定义Estimator模型并通过使用tf.estimator.train_and_evaluate完成训练和评估。

主要步骤：
> - 构建Estimator模型
> - 定义训练和测试过程中数据如何输入给模型
> - 定义传递给train_and_evaluate函数的训练、评估和导出的详述参数(TrainSpec and EvalSpec)
> - 使用tf.estimator.train_and_evaluate训练并评估模型

#### 构建Estimator
Estimator类提供了分布式模型训练和评估的内置支持，屏蔽了不同底层硬件平台（CPU、GPU、TPU）的差异。因此，建议大家总是使用Estimator来封装自己的模型。Tensorflow还提供了一些“Pre-made”的Estimator的子类可以用来高效地创建一些常用的标准模型，比如常用的“wide and deep”模型就可以用DNNLinearCombinedClassifier来创建。

Estimator的核心是模型函数（model function），模型函数构建训练、评估和预测用的计算图。当使用pre-made的Estimator时，模型函数已经为我们实现好了。当我们使用自定义的Estimator来创建自己的模型时，最重要的工作就是编写自己的模型函数。

模型函数的签名如下：
```python
def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration
```
前两个参数是输入函数(input_fn)返回的特性和标签，mode参数表明调用者是在训练、预测还是评估，params是其他一些自定义参数，通常是一个dict。

在模型函数内部，需要定义网络对应的计算图（graph)，并为模型在三个不同的阶段（训练、评估、预测）指定额外的操作，通过EstimatorSpec对象返回。
> - 在训练阶段返回的EstimatorSpec对象需要包含计算loss和最小化loss的操作（op）；
> - 在评估阶段返回的EstimatorSpec对象需要包含计算metrics的操作，和模型导出的操作；
> - 在预测阶段返回的EstimatorSpec对象需要包含跟获取预测结果的操作。

通常情况下，自己定义不同阶段的EstimatorSpec对象比较麻烦，这时可以用到另一个高阶API Head来帮忙简化开发任务。

为了理解下面模型函数，我们先看一下该项目的模型框架：![模型框架](https://pic4.zhimg.com/80/v2-66ed2ca5cec5ee35c4ea1e8cdb467453_720w.jpg)

整个模型函数的代码如下：
```python
def my_model(features, labels, mode, params):
  sentence = features['sentence']
  # Get word embeddings for each token in the sentence
  embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                               shape=[params["vocab_size"], FLAGS.embedding_size])
  sentence = tf.nn.embedding_lookup(embeddings, sentence) # shape:(batch, sentence_len, embedding_size)
  # add a channel dim, required by the conv2d and max_pooling2d method
  sentence = tf.expand_dims(sentence, -1) # shape:(batch, sentence_len/height, embedding_size/width, channels=1)

  pooled_outputs = []
  for filter_size in params["filter_sizes"]:
      conv = tf.layers.conv2d(
          sentence,
          filters=FLAGS.num_filters,
          kernel_size=[filter_size, FLAGS.embedding_size],
          strides=(1, 1),
          padding="VALID",
          activation=tf.nn.relu)
      pool = tf.layers.max_pooling2d(
          conv,
          pool_size=[FLAGS.sentence_max_len - filter_size + 1, 1],
          strides=(1, 1),
          padding="VALID")
      pooled_outputs.append(pool)
  h_pool = tf.concat(pooled_outputs, 3) # shape: (batch, 1, len(filter_size) * embedding_size, 1)
  h_pool_flat = tf.reshape(h_pool, [-1, FLAGS.num_filters * len(params["filter_sizes"])]) # shape: (batch, len(filter_size) * embedding_size)
  if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
    h_pool_flat = tf.layers.dropout(h_pool_flat, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
  logits = tf.layers.dense(h_pool_flat, FLAGS.num_classes, activation=None)

  optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
  
  def _train_op_fn(loss):
    return optimizer.minimize(loss, global_step=tf.train.get_global_step())

  my_head = tf.contrib.estimator.multi_class_head(n_classes=FLAGS.num_classes)
  return my_head.create_estimator_spec(
    features=features,
    mode=mode,
    labels=labels,
    logits=logits,
    train_op_fn=_train_op_fn
  )
```
#### 定义输入流 input pipeline
在Tensorflow中定义网络输入推荐使用Dataset API。

在本文的文本分类任务中，首先做一下预处理，把特殊符号和一些非字母类的其他文本内容去掉，然后构建完整的词汇表。词汇表用来把词映射到唯一的一个数字ID(函数index_table_from_file实现)。最终构建的训练样本的形式为<[word_id list], class_label>。

```python
def input_fn(path_csv, path_vocab, shuffle_buffer_size, num_oov_buckets):
  vocab = tf.contrib.lookup.index_table_from_file(path_vocab, num_oov_buckets=num_oov_buckets)
  # Load csv file, one example per line
  dataset = tf.data.TextLineDataset(path_csv)
  # Convert line into list of tokens, splitting by white space; then convert each token to an unique id
  dataset = dataset.map(lambda line: parse_line(line, vocab))
  if shuffle_buffer_size > 0:
    dataset = dataset.shuffle(shuffle_buffer_size).repeat()
  dataset = dataset.batch(FLAGS.batch_size).prefetch(1)
  return dataset
```

#### 模型的训练
定义好模型函数与输入函数之后，就可以用Estimator封装好分类器。同时需要定义estimator需要的TrainSpec和EvalSpec，把训练数据和评估数据喂给模型，这样就万事俱备了，最后只需要调用tf.estimator.train_and_evaluate就可以开始训练和评估模型了。
```python
classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
      'vocab_size': config["vocab_size"],
      'filter_sizes': map(int, FLAGS.filter_sizes.split(',')),
      'learning_rate': FLAGS.learning_rate,
      'dropout_rate': FLAGS.dropout_rate
    },
    config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
  )

  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: input_fn(path_train, path_words, FLAGS.shuffle_buffer_size, config["num_oov_buckets"]),
    max_steps=FLAGS.train_steps
  )
  input_fn_for_eval = lambda: input_fn(path_eval, path_words, 0, config["num_oov_buckets"])
  eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=300)

  tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
```