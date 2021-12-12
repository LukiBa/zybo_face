import numpy as np

import intuitus_nn as nn


def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = np.concatenate(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = np.concatenate(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = np.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = np.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = np.divide(inter_area, union_area, out=np.zeros_like(
        inter_area), where=union_area != 0, casting='unsafe')

    return iou


def xywh2xyxy(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def filter_boxes(x, conf_thres=0.1):
    x = x[x[:, 4] > conf_thres]
    if not x.shape[0]:
        return np.zeros((0, 4)), np.zeros((0, 1)), np.zeros((0, 1))

    # Compute conf
    x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

    boxes = xywh2xyxy(x[:, :4])
    classes = np.argmax(x[:, 5:], axis=-1)
    pred_conf = np.take_along_axis(
        x[:, 5:],
        np.expand_dims(classes, axis=-1),
        axis=-1)  # get values by index array

    return (boxes, pred_conf, classes)


def nms(boxes, pred_conf, classes, iou_threshold, sigma=0.3, score=0.1, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(classes))
    bboxes = np.concatenate((boxes, pred_conf, classes.reshape(classes.size, 1)), axis=-1)
    best_bboxes = []

    for clss in classes_in_img:
        clss_mask = (classes == clss)
        clss_bboxes = bboxes[clss_mask]

        while len(clss_bboxes) > 0:
            max_ind = np.argmax(clss_bboxes[:, 4])
            best_bbox = clss_bboxes[max_ind]
            # selcting all boxes except of best box
            clss_bboxes = np.concatenate([clss_bboxes[: max_ind], clss_bboxes[max_ind + 1:]])
            iou = bbox_iou(best_bbox[np.newaxis, :4].copy(), clss_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            elif method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            elif method == 'merge':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
                iou_boxes = np.concatenate((clss_bboxes, iou.reshape(iou.size, 1)), axis=-1)
                merge_boxes = iou_boxes[iou_mask]
                for i in range(merge_boxes.shape[0]):
                    best_bbox[:4] -= (best_bbox[:4] - merge_boxes[i, :4]
                                      )*(0.5*merge_boxes[i, 4]/best_bbox[4])
                    best_bbox[4] += merge_boxes[i, 4]*merge_boxes[i, 6]*sigma
            else:
                raise NotImplementedError(
                    'Non max surpression method :"' + method + '" is not implemented.')

            clss_bboxes[:, 4] = clss_bboxes[:, 4] * weight
            score_mask = clss_bboxes[:, 4] > 0.
            clss_bboxes = clss_bboxes[score_mask]
            if best_bbox[4] > score:
                best_bboxes.append(best_bbox)

    if len(best_bbox) > 0:
        return np.array(best_bboxes)
    return np.zeros((0, 6))


def yolov3_tiny_config():
    yolo_lb = {'anchors': np.array([[196, 210],  [224, 230],  [252, 270],  [280, 290], [307, 307]]),
               'stride': 32,
               'classes': 1}

    yolo_mb = {'anchors': np.array([[57, 62],  [85, 90],  [112, 120],  [140, 150], [168, 180]]),
               'stride': 16,
               'classes': 1}

    yolo_conf = {'lb': yolo_lb,
                 'mb': yolo_mb}

    return yolo_conf


def configure_Network(command_path, input_size):
    Net = nn.Sequential(command_path)
    buffer = Net.input(3, input_size, input_size)

    # %% backbone
    buffer = Net.conv2d(buffer, 16, (3, 3), max_pooling=True, command_file='conv2d_0.npz')
    buffer = Net.conv2d(buffer, 32, (3, 3), max_pooling=True, command_file='conv2d_1.npz')
    buffer = Net.conv2d(buffer, 64, (3, 3), max_pooling=True, command_file='conv2d_2.npz')
    buffer = Net.conv2d(buffer, 128, (3, 3), max_pooling=True, command_file='conv2d_3.npz')
    buffer = Net.conv2d(buffer, 256, (3, 3), command_file='conv2d_4.npz')
    route_1 = buffer
    buffer = Net.maxpool2d(buffer, strides=(2, 2))  # Software pooling (CPU)
    buffer = Net.conv2d(buffer, 512, (3, 3), command_file='conv2d_5.npz')
    buffer = Net.maxpool2d(buffer, strides=(1, 1))  # Software pooling (CPU)
    buffer = Net.conv2d(buffer, 1024, (3, 3), command_file='conv2d_6.npz')

    # # #%% head
    buffer = Net.conv2d(buffer, 256, (1, 1), command_file='conv2d_7.npz')

    lobj = buffer
    conv_lobj_branch = Net.conv2d(lobj, 512, (3, 3), command_file='conv2d_8.npz')
    conv_lbbox = Net.conv2d(conv_lobj_branch, 32, (1, 1), command_file='conv2d_9.npz')

    buffer = Net.conv2d(buffer, 128, (1, 1), command_file='conv2d_10.npz')
    buffer = Net.upsample(buffer)
    buffer = Net.concat(buffer, route_1)

    mobj = buffer
    conv_mobj_branch = Net.conv2d(mobj, 256, (3, 3), command_file='conv2d_11.npz')
    conv_mbbox = Net.conv2d(conv_mobj_branch, 32, (1, 1), command_file='conv2d_12.npz')

    Net.output(conv_lbbox)
    Net.output(conv_mbbox)

    Net.summary()
    return Net


class YoloLayer():
    def __init__(self, anchors: np.ndarray, num_classes: int, stride: int):
        self.anchors = anchors
        self.stride = stride  # layer stride
        self.na = len(anchors)  # number of anchors (3)
        self.nc = num_classes  # number of classes (80)
        self.no = num_classes + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.reshape(1, self.na, 1, 1, 2)
        self.grid = None
        print("na {}, no {}, ny {}, nx {}".format(self.na, self.no, self.nx, self.nx))

    def create_grids(self, ng=(13, 13)):
        self.ny, self.nx = ng  # y and x grid size
        self.ng = ng

        xv, yv = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        self.grid = np.stack((xv, yv), axis=-1).reshape(1, 1,
                                                        self.ny, self.ny, 2).astype(np.float32)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __call__(self, p):
        if not isinstance(self.grid, np.ndarray):
            _, ny, nx = p.shape  # 255, 13, 13
            self.create_grids((ny, nx))

        io = np.transpose(p.reshape(self.na, self.no, self.ny, self.nx), (0, 2, 3, 1)).copy('C')

        io[..., :2] = self.sigmoid(io[..., :2]) + self.grid  # xy
        io[..., 2:4] = np.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
        io[..., :4] *= self.stride
        io[..., 4:] = self.sigmoid(io[..., 4:])
        return io.reshape(-1, self.no)  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Detector():
    def __init__(self, commandPath, imgSize, confThreshold, iouThreshold, score):
        self.Net = configure_Network(commandPath, imgSize)
        yolo_config = yolov3_tiny_config()

        self.Yolo_lb = YoloLayer(yolo_config['lb']['anchors'],
                                 yolo_config['lb']['classes'],
                                 yolo_config['lb']['stride'])

        self.Yolo_mb = YoloLayer(yolo_config['mb']['anchors'],
                                 yolo_config['mb']['classes'],
                                 yolo_config['mb']['stride'])

        self.resultScale = 2**-4.0
        self.confThreshold = confThreshold
        self.iouThreshold = iouThreshold
        self.score = score

    def __call__(self, fmapIn):
        fmap_out = self.Net(fmapIn)
        pred_lb = self.Yolo_lb(fmap_out[0][:30, ...]*self.resultScale)
        pred_mb = self.Yolo_mb(fmap_out[1][:30, ...]*self.resultScale)
        inf_out = np.concatenate((pred_lb, pred_mb), axis=0)
        boxes, pred_conf, classes = filter_boxes(inf_out, self.confThreshold)
        if not boxes.shape[0]:
            return np.zeros([0, 4])
        best_bboxes = nms(boxes, pred_conf, classes, iou_threshold=self.iouThreshold,
                          score=self.score, method='merge')
        if not best_bboxes.shape[0]:
            return np.zeros([0, 4])
        return best_bboxes
    