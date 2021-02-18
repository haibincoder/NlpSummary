"""
nlpcda：一键中文数据增强工具，支持：
1.随机实体替换
2.近义词
3.近义近音字替换
4.随机字删除（内部细节：数字时间日期片段，内容不会删）
5.NER类 BIO 数据增强x
6.随机置换邻近的字：研表究明，汉字序顺并不定一影响文字的阅读理解<<是乱序的
7.中文等价字替换（1 一 壹 ①，2 二 贰 ②）
8.翻译互转实现的增强
9.使用simbert做生成式相似句生成
10.Cluster2Cluster生成更多样化的新数据
https://github.com/425776024/nlpcda
"""

from nlpcda import Randomword, Similarword, RandomDeleteChar

test_str = '中国平安创始人是谁'

smw = Randomword(create_num=3, change_rate=0.3)
rs1 = smw.replace(test_str)
print('随机实体替换>>>>>>')
for s in rs1:
    print(s)

test_str = '平安福有效期是多久'
smw = Similarword(create_num=3, change_rate=0.3)
rs1 = smw.replace(test_str)
print('随机同义词替换>>>>>>')
for s in rs1:
    print(s)

smw = RandomDeleteChar(create_num=3, change_rate=0.3)
rs1 = smw.replace(test_str)
print('随机字删除>>>>>>')
for s in rs1:
    print(s)
