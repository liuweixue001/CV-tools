import numpy as np
import xml.etree.ElementTree as ET
import glob
import random


def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])
    interarea = x * y
    clu_area = cluster[:, 0] * cluster[:, 1]
    box_area = box[0] * box[1]
    iou = interarea/(clu_area + box_area -interarea)
    return iou


def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    row = box.shape[0]
    # 初始化距离（iou）为极小值
    distance = np.empty((row, k))
    # 初始化每个box对应聚类中心的索引
    last_clu = np.zeros((row, ))
    np.random.seed()
    # 从box中随机选五个作为聚类中心
    cluster = box[np.random.choice(row, k, replace=False)]
    while True:
        # 计算每个box到五个聚类中心的距离
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)

        # 取出每个box相应的最小距离类的索引
        near = np.argmin(distance, axis=1)

        if (last_clu == near).all():
            break

        for j in range(k):
            # 找出near中到第j个聚类中心距离为最的所有索引
            # 根据索引找到相应的box，并对其取均值，更新聚类中心
            cluster[j] = np.median(box[near == j], axis=0)

        last_clu = near

        return cluster


def load_data(path):
    data = []
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        # 将实际长度转为比例
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            data.append([xmax-xmin, ymax-ymin])
    return np.array(data)


if __name__ == "__main__":
    anchors_nums = 5
    path = r"C:\Users\00769111\PycharmProjects\yolov5-master-changed\train_all\labels"
    data = load_data(path)
    out = kmeans(data, anchors_nums)
    out = out[np.argsort(out[:, 0])]
    print('acc:{:.2f}%'.format(avg_iou(data, out)*100))
    print(out*352)