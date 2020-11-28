import os
import pickle
import grpc
import numpy as np
from bert_base.server import set_logger

from bert_base.bert.extract_features import read_tokenized_examples, _truncate_seq_pair, read_line_examples,InputFeatures
from numpy import mat
import tensorflow as tf
from tensorflow.contrib.util import make_tensor_proto
import time


from tensorflow.contrib.crf import viterbi_decode
# from loader import input_from_line
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from bert import tokenization


# def input_from_line(line, max_seq_length, tag_to_id):
#     """
#     Take sentence data and return an input for
#     the training or the evaluation function.
#     """
#     string = [w[0].strip() for w in line]
#     # chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
#     #         for w in string]
#     char_line = ' '.join(string)  # 使用空格把汉字拼起来
#     text = tokenization.convert_to_unicode(char_line)
#
#     tags = ['O' for _ in string]
#
#     labels = ' '.join(tags)  # 使用空格把标签拼起来
#     labels = tokenization.convert_to_unicode(labels)
#
#     ids, mask, segment_ids, label_ids = convert_single_example(char_line=text,
#                                                                tag_to_id=tag_to_id,
#                                                                max_seq_length=max_seq_length,
#                                                                tokenizer=tokenizer,
#                                                                label_line=labels)
#     import numpy as np
#     segment_ids = np.reshape(segment_ids,(1, max_seq_length))
#     ids = np.reshape(ids, (1, max_seq_length))
#     mask = np.reshape(mask, (1, max_seq_length))
#     label_ids = np.reshape(label_ids, (1, max_seq_length))
#     return [string, segment_ids, ids, mask, label_ids]
#
#
#
#
# def convert(sentence):
#     maps_path = '/data/guoyin/ChineseNERTFServing/maps.pkl'
#     with open(maps_path, "rb") as f:
#         tag_to_id, id_to_tag = pickle.load(f)
#     return input_from_line(sentence,150,tag_to_id)
from bert.tokenization import FullTokenizer
def convert_lst_to_features(lst_str, seq_length, tokenizer, logger, is_tokenized=False, mask_cls_sep=False):
    """Loads a data file into a list of `InputBatch`s."""

    examples = read_tokenized_examples(lst_str) if is_tokenized else read_line_examples(lst_str)
    print(examples)


    _tokenize = lambda x: x if is_tokenized else tokenizer.tokenize(x)

    for (ex_index, example) in enumerate(examples):
        tokens_a = _tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = _tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        input_type_ids = [0] * len(tokens)
        input_mask = [int(not mask_cls_sep)] + [1] * len(tokens_a) + [int(not mask_cls_sep)]

        if tokens_b:
            tokens += tokens_b + ['[SEP]']
            input_type_ids += [1] * (len(tokens_b) + 1)
            input_mask += [1] * len(tokens_b) + [int(not mask_cls_sep)]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length. more pythonic
        pad_len = seq_length - len(input_ids)
        input_ids += [0] * pad_len
        input_mask += [0] * pad_len
        input_type_ids += [0] * pad_len

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        # logger.debug('tokens: %s' % ' '.join([tokenization.printable_text(x) for x in tokens]))
        # logger.debug('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
        # logger.debug('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
        # logger.debug('input_type_ids: %s' % ' '.join([str(x) for x in input_type_ids]))

        yield InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids)




channel = grpc.insecure_channel("10.0.43.58:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
##模型的名称，就是docker启动时的-e MODEL_NAME=tensorflowServingModels这个参数
request.model_spec.name = "tensorflowServingModels"
##在服务器上通过 saved_model_cli show --dir=‘linear_model/1’ --all进行查看获得
request.model_spec.signature_name ="classify"

start = time.perf_counter()
# 这里是向模型里面输入数据
sentence = ['重庆市南川中学副校长张竞说，寒假放假前高三年级还有10%左右的内容没复习完，学校开学受疫情影响延期不少，“老师们压力比较大，怕耽误复习进度。”']
is_tokenized = all(isinstance(el, list) for el in sentence)
tokenizer = FullTokenizer(vocab_file=os.path.join("D:\\LiuXianXian\\pycharm--code\\BertServiceBase_classify\\checkpoints", 'vocab.txt'))
logger = set_logger('WORKER-%d')

tmp_f = list(convert_lst_to_features(sentence, 128, tokenizer, logger,
                                                             is_tokenized, False))
ids =tmp_f[0].input_ids
mask=tmp_f[0].input_mask
segment_ids=tmp_f[0].input_type_ids
request.inputs['input_ids'].CopyFrom(make_tensor_proto([ids], dtype=np.int32))
request.inputs['input_mask'].CopyFrom(make_tensor_proto([mask], dtype=np.int32))
request.inputs['segment_ids'].CopyFrom(make_tensor_proto([segment_ids], dtype=np.int32))
# request.inputs['Dropout'].CopyFrom(make_tensor_proto(np.float32(1.0)))
response = stub.Predict(request, 30.0)
print(response)
end = time.perf_counter()
