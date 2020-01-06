# -*- coding: utf-8 -*-

"""

 @Time    : 2019/1/29 14:32
 @Author  : MaCan (ma_cancan@163.com)
 @File    : client_test.py
"""
import time
from bert_base.client import BertClient


def ner_test():
    with BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
        start_t = time.perf_counter()
        str1 = '1月24日，新华社对外发布了中央对雄安新区的指导意见，洋洋洒洒1.2万多字，17次提到北京，4次提到天津，信息量很大，其实也回答了人们关心的很多问题。'
        # rst = bc.encode([list(str1)], is_tokenized=True)
        # str1 = list(str1)
        rst = bc.encode([str1], is_tokenized=True)
        print('rst:', rst)
        print(len(rst[0]))
        print(time.perf_counter() - start_t)


def ner_cu_seg():
    """
    自定义分字
    :return:
    """
    with BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
        start_t = time.perf_counter()
        str1 = '1月24日，新华社对外发布了中央对雄安新区的指导意见，洋洋洒洒1.2万多字，17次提到北京，4次提到天津，信息量很大，其实也回答了人们关心的很多问题。'
        rst = bc.encode([list(str1)], is_tokenized=True)
        print('rst:', rst)
        print(len(rst[0]))
        print(time.perf_counter() - start_t)


def class_test():
    with BertClient(ip="10.0.46.99",show_server_config=False, check_version=False, check_length=False, mode='CLASS') as bc:
        start_t = time.perf_counter()
        str1 = '如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈'
        str2="茶树茶网蝽，Stephanitis chinensis Drake，属半翅目网蝽科冠网椿属的一种昆虫"
        str3="丝角蝗科，Oedipodidae，昆虫纲直翅目蝗总科的一个科"
        str4="爱德华·尼科·埃尔南迪斯（1986-），是一位身高只有70公分哥伦比亚男子，体重10公斤，只比随身行李高一些，2010年获吉尼斯世界纪录正式认证，成为全球当今最矮的成年男人"
        str5="《逐风行》是百度文学旗下纵横中文网签约作家清水秋风创作的一部东方玄幻小说，小说已于2014-04-28正式发布"
        str6="禅意歌者刘珂矣《一袖云》中诉知己…绵柔纯净的女声，将心中的万水千山尽意勾勒于这清素画音中"
        str7="《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧"
        str8="史雪梅，女，1962年生于陕西三原县，本科毕业，是咸阳市妇女书画协会副会长，就职于陕西省宝鸡峡管理局"
        rst = bc.encode([str1,str2,str3,str4,str5,str6,str7,str8])
        print('rst:', rst)
        print('time used:{}'.format(time.perf_counter() - start_t))

def simi_test():
    with BertClient(show_server_config=False, check_version=False, check_length=False, mode='CLASS') as bc:
        start_t = time.perf_counter()
        str1 = '我想开通花呗'
        str2 = '我也想开通花呗'
        str3="你好"
        str4="你好呀"
        sss=str3+"|||"+str4


        ss=str1+"|||"+str2
        rst = bc.encode([ss,sss])
        print('rst:', rst)
        print('time used:{}'.format(time.perf_counter() - start_t))


if __name__ == '__main__':
    class_test()
    # simi_test()
    # ner_test()
    # ner_cu_seg()