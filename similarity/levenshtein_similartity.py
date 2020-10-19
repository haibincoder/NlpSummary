def min_distance(q1, q2):
    len1 = len(q1)
    len2 = len(q2)
    diff = [[0 for i in range(len2+1)] for j in range(len1+1)]

    for i in range(0, len1+1):
        diff[i][0] = i
    for i in range(0, len2+1):
        diff[0][i] = i
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            temp = 0 if q1[i-1] == q2[j-1] else 1
            diff[i][j] = min(diff[i-1][j-1] + temp, diff[i][j-1] + 1, diff[i-1][j] + 1)
    return diff[len1][len2]


def similarity(q1, q2):
    if q1 is None or q2 is None:
        return 0
    if q1 == q2:
        return 1
    distance = min_distance(q1, q2)
    return 1 - (distance / max(len(q1), len(q2)))


if __name__ == "__main__":
    q1 = "怎么用百度"
    q2 = "怎么登陆百度"
    print(f'scord: {similarity(q1, q2)}')
