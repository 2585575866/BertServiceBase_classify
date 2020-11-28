#!/usr/bin/env python
# encoding: utf-8
# @Time    : 2020/6/15 13:10
# @Author  : lxx
# @File    : test1.py
# @Software: PyCharm

import tensorflow as tf
from bert_base.bert.extract_features import convert_lst_to_features

from bert_base.server import set_logger

from bert_base.bert.tokenization import FullTokenizer

from bert import modeling
import os


def create_classification_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
    """
    :param bert_config:
    :param is_training:
    :param input_ids:
    :param input_mask:
    :param segment_ids:
    :param labels:
    :param num_labels:
    :param use_one_hot_embedding:
    :return:
    """

    # import tensorflow as tf
    # import modeling

    # 通过传入的训练数据，进行representation
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
    )

    embedding_layer = model.get_sequence_output()
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    # predict = CNN_Classification(embedding_chars=embedding_layer,
    #                                labels=labels,
    #                                num_tags=num_labels,
    #                                sequence_length=FLAGS.max_seq_length,
    #                                embedding_dims=embedding_layer.shape[-1].value,
    #                                vocab_size=0,
    #                                filter_sizes=[3, 4, 5],
    #                                num_filters=3,
    #                                dropout_keep_prob=FLAGS.dropout_keep_prob,
    #                                l2_reg_lambda=0.001)
    # loss, predictions, probabilities = predict.add_cnn_layer()

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.sigmoid(logits)
        if labels is not None:
            label_ids = tf.cast(labels, tf.float32)
            per_example_loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_ids), axis=-1)
            loss = tf.reduce_mean(per_example_loss)
        else:
            loss, per_example_loss = None, None
    return (loss, per_example_loss, logits, probabilities)



def ckpt2pb():

    max_seq_len = 128
    num_labels = 49
    input_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_ids')
    input_mask = tf.placeholder(tf.int32, (None, max_seq_len), 'input_mask')
    segment_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'segment_ids')

    bert_config = modeling.BertConfig.from_json_file(os.path.join("D:\\LiuXianXian\\Pycharm--code\\BertServiceBase_classify\\checkpoints", 'bert_config.json'))

    loss, per_example_loss, logits, probabilities = create_classification_model(bert_config=bert_config,
                                                                                                is_training=False,
                                                                                                input_ids=input_ids,
                                                                                                input_mask=input_mask,
                                                                                                segment_ids=segment_ids,
                                                                                                labels=None,
                                                                                                num_labels=num_labels)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state("D:\\LiuXianXian\\Pycharm--code\\BertServiceBase_classify\\mypoints")
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        builder = tf.saved_model.builder.SavedModelBuilder(
            "D:\\LiuXianXian\\Pycharm--code\\BertServiceBase_classify\\pbs")
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                'input_ids': tf.saved_model.utils.build_tensor_info(input_ids),
                'input_mask': tf.saved_model.utils.build_tensor_info(input_mask),
                'segment_ids': tf.saved_model.utils.build_tensor_info(segment_ids),
            },
            outputs={
                'pred_prob': tf.saved_model.utils.build_tensor_info(probabilities),
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'classify':
                    prediction_signature,
            },
            clear_devices=False,
            legacy_init_op=legacy_init_op)

        builder.save()
        print("model export done.")

# 查看导出的模型的结构
def get_model_structure():
    export_dir = 'D:\\LiuXianXian\\Pycharm--code\\BertServiceBase_classify\\pbs'
    sess = tf.Session()
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    signature = meta_graph_def.signature_def
    print(signature['classify'])

def predict(sess,feed_dict,signature):
    ids_tensor_name = signature['classify'].inputs['input_ids'].name
    mask_tensor_name = signature['classify'].inputs['input_mask'].name
    segment_tensor_name = signature['classify'].inputs['segment_ids'].name
    # Dropout_tensor_name = signature['classify'].inputs['Dropout'].name

    # lengths_tensor_name = signature['classify'].outputs['lengths'].name
    logits_tensor_name = signature['classify'].outputs['pred_prob'].name
    # trans_tensor_name = signature['classify'].outputs['trans'].name

    input_ids = sess.graph.get_tensor_by_name(ids_tensor_name)
    input_mask = sess.graph.get_tensor_by_name(mask_tensor_name)
    segment_ids = sess.graph.get_tensor_by_name(segment_tensor_name)
    # Dropout = sess.graph.get_tensor_by_name(Dropout_tensor_name)

    # lengths = sess.graph.get_tensor_by_name(lengths_tensor_name)
    logits = sess.graph.get_tensor_by_name(logits_tensor_name)
    # trans = sess.graph.get_tensor_by_name(trans_tensor_name)
    # 注意，这里的feed_dict要和模型里面的键值一样，否则会报错的
    logits= sess.run([logits], feed_dict={input_ids:feed_dict['input_ids'],
                                                           input_mask:feed_dict['input_mask'],
                                                           segment_ids:feed_dict['segment_ids']
                                                           })
    return logits

def localTestPb():
    sentence = ['重庆市南川中学副校长张竞说，寒假放假前高三年级还有10%左右的内容没复习完，学校开学受疫情影响延期不少，“老师们压力比较大，怕耽误复习进度。”']
    is_tokenized = all(isinstance(el, list) for el in sentence)
    tokenizer = FullTokenizer(
        vocab_file=os.path.join("D:\\LiuXianXian\\pycharm--code\\BertServiceBase_classify\\checkpoints", 'vocab.txt'))
    logger = set_logger('WORKER-%d')
    tmp_f = list(convert_lst_to_features(sentence, 128, tokenizer, logger, is_tokenized, False))
    ids = tmp_f[0].input_ids
    mask = tmp_f[0].input_mask
    segment_ids = tmp_f[0].input_type_ids

    export_dir = 'D:\\LiuXianXian\\Pycharm--code\\BertServiceBase_classify\\pbs'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    signature = meta_graph_def.signature_def
    feed_dict = {"input_ids": [ids], "input_mask": [mask], "segment_ids": [segment_ids]}
    logits = predict(sess,feed_dict=feed_dict, signature=signature)
    print(logits)

#转换
# ckpt2pb()

#查看模型结构
get_model_structure()


#本地测试导出的pb模型
# localTestPb()