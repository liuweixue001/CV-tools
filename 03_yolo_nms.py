import numpy as np


def non_max_suppression(boxes, num_classes, conf_thres=0.5, nms_thres=0.4,sigma=0.5):
    # boxes为 batch_size, all_anchors, 4+1+num_cls
    # 取出batch_size
    bs = np.shape(boxes)[0]
    # 将框从偏离中心点，长宽转为左上角、右下角的形式
    shape_boxes = np.zeros_like(boxes[:, :, 4])
    shape_boxes[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2] / 2
    shape_boxes[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3] / 2
    shape_boxes[:, :, 2] = boxes[:, :, 0] - boxes[:, :, 2] / 2
    shape_boxes[:, :, 3] = boxes[:, :, 1] - boxes[:, :, 4] / 2

    boxes[:, :, :4] = shape_boxes
    out_put = []
    # 遍历每一张图片
    for i in range(bs):
        prediction = boxes[i]
        # 筛选出每一张图片中置信度大于阈值的框
        score = prediction[:, 4]
        mask = score > conf_thres
        detections = prediction[mask]
        class_conf = np.expand_dims(np.max(detections[:, 5:], axis=-1), axis=-1)
        class_pred = np.expand_dims(np.argmax(detections[:, 5:], axis=-1), axis=-1)
        detections = np.concatenate([detections[:, :5], class_conf, class_pred], axis=-1)
        # 得到所有类别
        unique_class = np.unique(detections[:, -1])
        if len(unique_class) == 0:
            continue

        best_box = []
        # 遍历每一个类别
        for c in unique_class:
            cls_mask = detections[:, -1] == c
            # 筛选出属于c类的框
            detection = detections[cls_mask]
            # 获取输入c类的框的概率，并排序
            scores = detection[:, 4]
            arg_sort = np.argsort(scores)[::-1]
            detection = detection[arg_sort]

            # 根据概率最大，iou＜阈值，多次筛选
            # 最终应为每个类的每个目标对应一个检测框
            while len(detection) != 0:
                best_box.append(detection[0])
                if len(detection) == 1:
                    break
                # 计算其他预测框与最大概率预测框的iou，并将iou>阈值的预测框删除
                ious = iou(best_box[-1], detection[1:])
                # 普通非极大抑制
                detection = detection[1:][ious<nms_thres]
                # 柔性非极大抑制，用于处理两个同类别目标重叠较大的情况
                # 需重新排序
                # detection[1:, 4] = np.exp(-(ious * ious)/sigma)*detection[1:, 4]
                # detection = detection[1:]
                # scores = detection[:, 4]
                # arg_sort = np.argsort(scores)[::-1]
                # detection = detection[arg_sort]
            out_put.append(best_box)
        return out_put


def iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[0], b2[1], b2[2], b2[3]
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.maximum(b1_x2, b2_x2)
    inter_rect_y2 = np.maximum(b1_y2, b2_y2)
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    area_b1 = (b1_x2 - b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    iou = inter_area/(area_b1 + area_b2 - inter_area)
    return iou