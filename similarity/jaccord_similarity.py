def jaccord_similarity(q1, q2):
    set1 = set(q1)
    set2 = set(q2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection)/len(union)


if __name__ == "__main__":
    q1 = "怎么用百度"
    q2 = "怎么登陆百度"
    print(f'score:{jaccord_similarity(q1, q2)}')

