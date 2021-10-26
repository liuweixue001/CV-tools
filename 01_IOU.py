import cv2
import numpy as np


def CountIOU(RectA, RectB):
    xA = max(RectA[0], RectB[0])
    yA = max(RectA[1], RectB[1])
    xB = min(RectA[2], RectB[2])
    yB = min(RectA[3], RectB[3])
    # 计算交集面积，当无交集时(yB-yA)与(xB-xA)为负值
    interArea = max(0, yB-yA+1) * max(0, xB-xA+1)
    RecA_Area = (RectA[2]-RectA[0])*(RectA[3]-RectA[1])
    RecB_Area = (RectB[2] - RectB[0]) * (RectB[3] - RectB[1])
    IOU = interArea/float(RecA_Area + RecB_Area - interArea)
    return IOU


img = np.zeros((512, 512, 3), np.uint8) + 255

RectA = [50, 50, 300, 300]
RectB = [60, 60, 320, 320]


cv2.rectangle(img, (RectA[0], RectA[1]), (RectA[2], RectA[3]), (0, 255, 0), 5)
cv2.rectangle(img, (RectB[0], RectB[1]), (RectB[2], RectB[3]), (255, 0, 0), 5)

iou = CountIOU(RectA, RectB)
cv2.putText(img, "IOU = %.2F"%iou, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (175, 0, 175), 2)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()