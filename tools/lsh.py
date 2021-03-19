from lshash import LSHash

"""
利用局部敏感hash查找近似值
"""

lsh = LSHash(6, 8)
lsh.index([1,2,3,4,5,6,7,8])
lsh.index([2,3,4,5,6,7,8,9])
lsh.index([10,12,99,1,5,31,2,3])
lsh.query([1,2,3,4,5,6,7,7])