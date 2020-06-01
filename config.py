
import tensorflow as tf
import json
import os

#结巴分词自定义词典路径
cut_words_dict_path = "resource/my_dict.txt"

#bert配置
data_root = "/home/zhangmeiwei/pre_models/bert/chinese_L-12_H-768_A-12/"
bert_config_file = data_root + "bert_config.json"
init_checkpoint = data_root + "bert_model.ckpt"
bert_vocab_file = data_root + "vocab.txt"

#doc2vec配置文件
doc2vec_model = "resource/models/doc2vec/doc_vector.model"

is_develop = json.load(open("./branch.json"))["is_develop"] #开发环境还是发布环境

if is_develop:
    '''
    发布环境
    '''
    pass
else:
    '''
    开发环境
    '''
    pass

max_clustering_sim = 0.95
ap_damping = 0.8

gpu_path = os.veriron['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(allow_growth=True)

stopwords_path = "/home/zhangmeiwei/static_model_online/medical_record_cluster/stopwords.txt"
sentence_path = "/home/zhangmeiwei/static_model_online/medical_record_cluster/sentence.txt"
corpus_path = "/home/zhangmeiwei/static_model_online/medical_record_cluster/medical_true.txt"
w2v_path = "/home/zhangmeiwei/static_model_online/medical_record_cluster/w2v.model"
fre_path = "/home/zhangmeiwei/static_model_online/medical_record_cluster/word_freq.pkl"








