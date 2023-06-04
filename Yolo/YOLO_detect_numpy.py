import colorsys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image
from torchvision.ops import nms


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_anchors(anchors_path):
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def preprocess_input(image):
    image /= 255.0
    return image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cpu_nms(dets, score, thresh, original_size):
    x1 = dets[:, 0] * original_size
    y1 = dets[:, 1] * original_size
    x2 = dets[:, 2] * original_size
    y2 = dets[:, 3] * original_size
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = score
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap
        print(w)
        print(h)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        print(ious)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep

class YOLO():
    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self):
        self.classes_path = "./coco_classes.txt"
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors_path = "./yolo_anchors.txt"
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.input_shape = [416, 416]
        self.anchors_mask = [[3, 4, 5], [0, 1, 2]]
        self.bbox_attrs = 5 + self.num_classes
        self.model_path = "./yolov3_tiny_model.pth"
        self.cuda = True
        self.letterbox_image = True
        self.confidence = 0.14
        self.nms_iou = 0.1

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self):
        # ---------------------------------------------------#
        #   建立yolov3模型，载入yolov3模型的权重
        # ---------------------------------------------------#
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = torch.load(self.model_path, map_location=device)
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            start = time.time()
            outputs = self.net(images)
            end = time.time()
            print(end-start)
            outputs = self.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.non_max_suppression(np.concatenate(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            print(i)
            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 255, 13, 13
            #   batch_size, 255, 26, 26
            #   batch_size, 255, 52, 52
            #-----------------------------------------------#
            input = np.array(input.cpu())
            batch_size      = input.shape[0]
            input_height    = input.shape[2]
            input_width     = input.shape[3]

            #-----------------------------------------------#
            #   输入为416x416时
            #   stride_h = stride_w = 32、16、8
            #-----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            #-------------------------------------------------#
            #   此时获得的scaled_anchors大小是相对于特征层的
            #-------------------------------------------------#
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h)
                              for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 13, 13, 85
            #   batch_size, 3, 26, 26, 85
            #   batch_size, 3, 52, 52, 85
            #-----------------------------------------------#
            prediction = input.reshape(batch_size, len(self.anchors_mask[i]),
                                       self.bbox_attrs, input_height, input_width).transpose(0, 1, 3, 4, 2)

            #-----------------------------------------------#
            #   先验框的中心位置的调整参数
            #-----------------------------------------------#
            x = sigmoid(prediction[..., 0])
            y = sigmoid(prediction[..., 1])
            #-----------------------------------------------#
            #   先验框的宽高调整参数
            #-----------------------------------------------#
            w = prediction[..., 2]
            h = prediction[..., 3]
            #-----------------------------------------------#
            #   获得置信度，是否有物体
            #-----------------------------------------------#
            conf        = sigmoid(prediction[..., 4])
            #-----------------------------------------------#
            #   种类置信度
            #-----------------------------------------------#
            pred_cls    = sigmoid(prediction[..., 5:])


            #----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角
            #   batch_size,3,13,13
            #----------------------------------------------------------#
            grid_x = np.linspace(0, input_width - 1, input_width).reshape(1, -1).repeat(input_height, axis = 0)[None]\
                .repeat(batch_size * len(self.anchors_mask[i]), 0).reshape(x.shape)
            grid_y = np.linspace(0, input_height - 1, input_height).reshape(1, -1).repeat(input_width, 0).transpose()[None]\
                .repeat(batch_size * len(self.anchors_mask[i]), 0).reshape(y.shape)

            #----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,13,13
            #----------------------------------------------------------#
            anchor_w = np.array(scaled_anchors)[:,0].reshape(-1, 1)
            anchor_h = np.array(scaled_anchors)[:,1].reshape(-1, 1)
            #anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            #anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
            anchor_w = anchor_w[None].repeat(batch_size, 0)[None].repeat(input_height * input_width, 2).reshape(w.shape)
            anchor_h = anchor_h[None].repeat(batch_size, 0)[None].repeat(input_height * input_width, 2).reshape(h.shape)

            #----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #----------------------------------------------------------#
            pred_boxes          = np.zeros(prediction[..., :4].shape)
            pred_boxes[..., 0]  = x.data + grid_x
            pred_boxes[..., 1]  = y.data + grid_y
            pred_boxes[..., 2]  = np.exp(w.data) * anchor_w
            pred_boxes[..., 3]  = np.exp(h.data) * anchor_h

            #----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            #----------------------------------------------------------#
            _scale = np.array([input_width, input_height, input_width, input_height])
            output = np.concatenate((pred_boxes.reshape(batch_size, -1, 4) / _scale,
                                conf.reshape(batch_size, -1, 1), pred_cls.reshape(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                            nms_thres=0.4):
        # ----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        # ----------------------------------------------------------#
        box_corner = np.zeros(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # ----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            # ----------------------------------------------------------#
            #class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            class_conf = np.max(image_pred[:, 5:5 + num_classes], 1).reshape(-1, 1)
            class_pred = np.argmax(image_pred[:, 5:5 + num_classes], 1).reshape(-1, 1)

            # ----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            # ----------------------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            # ----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            # ----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_conf = class_conf.astype(float)
            class_pred = class_pred[conf_mask]
            class_pred = class_pred.astype(float)
            if not image_pred.shape[0]:
                continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            # -------------------------------------------------------------------------#
            detections = np.concatenate((image_pred[:, :5], class_conf , class_pred), 1)

            # ------------------------------------------#
            #   获得预测结果中包含的所有种类
            # ------------------------------------------#
            unique_labels = np.unique(detections[:, -1])

            for c in unique_labels:
                # ------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                # ------------------------------------------#
                detections_class = detections[detections[:, -1] == c]

                # ------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                # ------------------------------------------#
                #print(detections_class[:, :4])
                #print(detections_class[:, 4] * detections_class[:, 5])
                keep = cpu_nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_thres,
                    self.input_shape[0] * self.input_shape[1]
                )
                #print(keep)
                max_detections = detections_class[keep]

                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data

                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))

            if output[i] is not None:
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

if __name__ == "__main__":
    yolo = YOLO()
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()