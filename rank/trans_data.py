import sys


def save_data(group_data, output_feature, output_group):
    if len(group_data) == 0:
        return

    output_group.write(str(len(group_data))+"\n")
    for data in group_data:
        # only include nonzero features
        feats = [ p for p in data[2:] if float(p.split(':')[1]) != 0.0 ]
        output_feature.write(data[0] + " " + " ".join(feats) + "\n")


def trans(input, output_feature_path, output_group_path):
    # if len(sys.argv) != 4:
    #     print("Usage: python trans_data.py [Ranksvm Format Input] [Output Feature File] [Output Group File]")
    #     sys.exit(0)
    #
    # fi = open(sys.argv[1])
    output_feature = open(output_feature_path, 'w')
    output_group = open(output_group_path, 'w')
    fi = open(input)

    group_data = []
    group = ""
    for line in fi:
        if not line:
            break
        if "#" in line:
            line = line[:line.index("#")]
        splits = line.strip().split(" ")
        if splits[1] != group:
            save_data(group_data,output_feature,output_group)
            group_data = []
        group = splits[1]
        group_data.append(splits)

    save_data(group_data,output_feature,output_group)

    fi.close()
    output_feature.close()
    output_group.close()


if __name__=="__main__":
    path = "mq2008_fold1"
    trans(path+"/train.txt", "mq2008.train", "mq2008.train.group")
    trans(path + "/test.txt", "mq2008.test", "mq2008.test.group")
    trans(path + "/vali.txt", "mq2008.vali", "mq2008.vali.group")
    print("finish")