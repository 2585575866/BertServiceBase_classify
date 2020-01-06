#!/usr/bin/env python
# encoding: utf-8
# @Time    : 2019/12/17 12:52
# @Author  : lxx
# @File    : test_lxx.py
# @Software: PyCharm

import re




def test():
    str1 = '我的花呗为什么不支持交易'
    str2 = '你好'
    line = str1 + "|||" + str2
    m = re.match(r"^(.*)\|\|\|(.*)$", line)
    if m is None:
        text_a = line
        print("text_a====" + text_a)
    else:
        text_a = m.group(1)
        text_b = m.group(2)
        print("text_a===="+text_a)
        print("text_b===="+text_b)

if __name__ == "__main__":
    test()