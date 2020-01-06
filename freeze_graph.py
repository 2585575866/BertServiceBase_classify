#!/usr/bin/env python
# encoding: utf-8
# @Time    : 2019/12/15 12:52
# @Author  : lxx
# @File    : freeze_graph.py
# @Software: PyCharm


# !/usr/bin/env python3
# coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

'''
BERT模型文件 ckpt转pb 工具
'''

# import contextlib
import json
import os
from enum import Enum
from termcolor import colored
import sys
import modeling
import logging
import tensorflow as tf
import argparse
import pickle


def set_logger(context, verbose=False):
    if os.name == 'nt':  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s:%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s:%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s:%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s:%s' % (self.context, msg), flush=True)


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


def create_ner_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                     num_token_labels, num_predicate_labels,max_seq_length):
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

    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token. float Tensor of shape [batch_size, hidden_size]
    predicate_output_layer = model.get_pooled_output()

    intent_hidden_size = predicate_output_layer.shape[-1].value

    predicate_output_weights = tf.get_variable(
        "predicate_output_weights", [num_predicate_labels, intent_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    predicate_output_bias = tf.get_variable(
        "predicate_output_bias", [num_predicate_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("predicate_loss"):
        if is_training:
            # I.e., 0.1 dropout
            predicate_output_layer = tf.nn.dropout(predicate_output_layer, keep_prob=0.9)

        predicate_logits = tf.matmul(predicate_output_layer, predicate_output_weights, transpose_b=True)
        predicate_logits = tf.nn.bias_add(predicate_logits, predicate_output_bias)
        predicate_probabilities = tf.nn.softmax(predicate_logits, axis=-1)
        predicate_prediction = tf.argmax(predicate_probabilities, axis=-1, output_type=tf.int32)
        # predicate_labels = tf.one_hot(predicate_label_id, depth=num_predicate_labels, dtype=tf.float32)
        # predicate_per_example_loss = tf.reduce_sum(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=predicate_logits, labels=predicate_labels), -1)
        # predicate_loss = tf.reduce_mean(predicate_per_example_loss)

    #     """Gets final hidden layer of encoder.
    #
    #     Returns:
    #       float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
    #       to the final hidden of the transformer encoder.
    #     """
    token_label_output_layer = model.get_sequence_output()

    token_label_hidden_size = token_label_output_layer.shape[-1].value

    token_label_output_weight = tf.get_variable(
        "token_label_output_weights", [num_token_labels, token_label_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    token_label_output_bias = tf.get_variable(
        "token_label_output_bias", [num_token_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("token_label_loss"):
        if is_training:
            token_label_output_layer = tf.nn.dropout(token_label_output_layer, keep_prob=0.9)
        token_label_output_layer = tf.reshape(token_label_output_layer, [-1, token_label_hidden_size])
        token_label_logits = tf.matmul(token_label_output_layer, token_label_output_weight, transpose_b=True)
        token_label_logits = tf.nn.bias_add(token_label_logits, token_label_output_bias)

        token_label_logits = tf.reshape(token_label_logits, [-1, max_seq_length, num_token_labels])
        # token_label_log_probs = tf.nn.log_softmax(token_label_logits, axis=-1)
        # token_label_one_hot_labels = tf.one_hot(token_label_ids, depth=num_token_labels, dtype=tf.float32)
        # token_label_per_example_loss = -tf.reduce_sum(token_label_one_hot_labels * token_label_log_probs, axis=-1)
        # token_label_loss = tf.reduce_sum(token_label_per_example_loss)
        token_label_probabilities = tf.nn.softmax(token_label_logits, axis=-1)
        # token_label_predictions = tf.argmax(token_label_probabilities, axis=-1)
        # return (token_label_loss, token_label_per_example_loss, token_label_logits, token_label_predict)

    # loss = 0.5 * predicate_loss + token_label_loss
    # return (loss,
    #         predicate_loss, predicate_per_example_loss, predicate_probabilities, predicate_prediction,
    #         token_label_loss, token_label_per_example_loss, token_label_logits, token_label_predictions)
    return (predicate_probabilities,token_label_probabilities)


def init_predict_var(path):
    num_labels = 2
    label2id = None
    id2label = None
    label2id_file = os.path.join(path, 'label2id.pkl')
    if os.path.exists(label2id_file):
        with open(label2id_file, 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
            num_labels = len(label2id.items())
        print('num_labels:%d' % num_labels)
    else:
        print('Can\'t found %s' % label2id_file)
    return num_labels, label2id, id2label

def optimize_ner_model(args, num_labels,  logger=None):
    """
    加载中文NER模型
    :param args:
    :param num_labels:
    :param logger:
    :return:
    """
    if not logger:
        logger = set_logger(colored('NER_MODEL, Lodding...', 'cyan'), args.verbose)
    try:
        # 如果PB文件已经存在则，返回PB文件的路径，否则将模型转化为PB文件，并且返回存储PB文件的路径
        if args.model_pb_dir is None:
            # 获取当前的运行路径
            tmp_file = os.path.join(os.getcwd(), 'predict_optimizer')
            if not os.path.exists(tmp_file):
                os.mkdir(tmp_file)
        else:
            tmp_file = args.model_pb_dir
        pb_file = os.path.join(tmp_file, 'ner_model.pb')
        if os.path.exists(pb_file):
            print('pb_file exits', pb_file)
            return pb_file

        import tensorflow as tf

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
                input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')
                segment_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'segment_ids')
                bert_config = modeling.BertConfig.from_json_file(os.path.join(args.bert_model_dir, 'bert_config.json'))
                from bert_base.train.models import create_model
                (predicate_probabilities,token_label_probabilities) = create_ner_model(
                    bert_config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                    num_token_labels=args.num_token_labels,num_predicate_labels=args.num_predicate_labels,max_seq_length=args.max_seq_length)
                predicate_probabilities = tf.identity(predicate_probabilities, 'predicate_probabilities')
                token_label_probabilities = tf.identity(token_label_probabilities, 'token_label_probabilities')
                saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))
                logger.info('freeze...')
                from tensorflow.python.framework import graph_util
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['predicate_probabilities',"token_label_probabilities"])
                logger.info('model cut finished !!!')
        # 存储二进制模型到文件中
        logger.info('write graph to a tmp file: %s' % pb_file)
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return pb_file
    except Exception as e:
        logger.error('fail to optimize the graph! %s' % e, exc_info=True)




def optimize_class_model(args, logger=None):
    """
    加载中文分类模型
    :param args:
    :param num_labels:
    :param logger:
    :return:
    """

    if not logger:
        logger = set_logger(colored('CLASSIFICATION_MODEL, Lodding...', 'cyan'), args.verbose)
        pass
    try:
        # 如果PB文件已经存在则，返回PB文件的路径，否则将模型转化为PB文件，并且返回存储PB文件的路径
        if args.model_pb_dir is None:
            tmp_dir = args.model_dir
        else:
            tmp_dir = args.model_pb_dir

        pb_file = os.path.join(tmp_dir, 'classification_model.pb')
        if os.path.exists(pb_file):
            print('pb_file exits', pb_file)
            return pb_file

        # 增加 从label2id.pkl中读取num_labels, 这样也可以不用指定num_labels参数； 2019/4/17
        if not args.classify_num_labels:
            num_labels, label2id, id2label = init_predict_var(tmp_dir)
        else:
            num_labels = args.classify_num_labels
        # ---

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
                input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')
                segment_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'segment_ids')

                bert_config = modeling.BertConfig.from_json_file(os.path.join(args.bert_model_dir, 'bert_config.json'))

                loss, per_example_loss, logits, probabilities = create_classification_model(bert_config=bert_config,
                                                                                            is_training=False,
                                                                                            input_ids=input_ids,
                                                                                            input_mask=input_mask,
                                                                                            segment_ids=segment_ids,
                                                                                            labels=None,
                                                                                            num_labels=num_labels)

                # pred_ids = tf.argmax(probabilities, axis=-1, output_type=tf.int32, name='pred_ids')
                # pred_ids = tf.identity(pred_ids, 'pred_ids')

                probabilities = tf.identity(probabilities, 'pred_prob')
                saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                latest_checkpoint = tf.train.latest_checkpoint(args.model_dir)
                logger.info('loading... %s ' % latest_checkpoint)
                saver.restore(sess, latest_checkpoint)
                logger.info('freeze...')
                from tensorflow.python.framework import graph_util
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['pred_prob'])
                logger.info('predict cut finished !!!')

        # 存储二进制模型到文件中
        logger.info('write graph to a tmp file: %s' % pb_file)
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return pb_file
    except Exception as e:
        logger.error('fail to optimize the graph! %s' % e, exc_info=True)


if __name__ == '__main__':
    pass

    """
    bert_model_dir="/mnt/sda1/transdat/bert-demo/bert/chinese_L-12_H-768_A-12"
    model_dir="/mnt/sda1/transdat/bert-demo/bert/output/demo7"
    model_pb_dir=model_dir 
    max_seq_len=128
    num_labels=2
    """

    parser = argparse.ArgumentParser(description='Trans ckpt file to .pb file')
    parser.add_argument('-bert_model_dir', type=str, required=True,
                        help='chinese google bert model path')
    parser.add_argument('-model_dir', type=str, required=True,
                        help='directory of a pretrained BERT model')
    parser.add_argument('-model_pb_dir', type=str, default=None,
                        help='directory of a pretrained BERT model,default = model_dir')
    parser.add_argument('-max_seq_len', type=int, default=128,
                        help='maximum length of a sequence,default:128')
    parser.add_argument('-classify_num_labels', type=int, default=None,
                        help='length of classity labels,default=2')

    parser.add_argument('-ner_num_labels', type=int, default=None,
                        help='length of ner labels,default=2')
    parser.add_argument('-verbose', action='store_true', default=False,
                        help='turn on tensorflow logging for debug')

    args = parser.parse_args()

    optimize_class_model(args, logger=None)
    # optimize_ner_model(args, logger=None)