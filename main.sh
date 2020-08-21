#!/usr/bin/env bash


python D:/fenci_jieba/testf_result/jieba_only.py
for ((key=8;key<30;key++));
do


python D:/fenci_jieba/util/clean.py
python D:/fenci_jieba/make_cidian/word2vec_make_cidian2.py --i 1 --key $key
python D:/fenci_jieba/make_cidian/the_last_cidian.py --num 4
python D:/fenci_jieba/testf_result/jieba_add_cidian.py


done
for ((num=0;num<30;num++));
do

python D:/fenci_jieba/util/clean.py
python D:/fenci_jieba/make_cidian/word2vec_make_cidian2.py --i 1 --key 13
python D:/fenci_jieba/make_cidian/the_last_cidian.py --num $num
python D:/fenci_jieba/testf_result/jieba_add_cidian.py

done





