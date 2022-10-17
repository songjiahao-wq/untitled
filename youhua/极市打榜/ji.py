import json
import torch
import sys
import numpy as np
import cv2
from pathlib import Path

# from ensemble_boxes import weighted_boxes_fusion
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox

device = torch.device("cuda:0")
model_path = r'/project/train/models/exp/weights/best.pt'  # 模型地址一定要和测试阶段选择的模型地址一致！！！
@torch.no_grad()
def init():
    weights = model_path
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or

    half = True  # use FP16 half-precision inference

    device = select_device(device)
    w = str(weights[0] if isinstance(weights, list) else weights)
    model = attempt_load(weights, map_location=device)
    if half:
        model.half()  # to FP16
    model.eval()
    return model


def process_image(handle=None, input_image=None, args=None, **kwargs):
    with torch.no_grad():
        half = True  # use FP16 half-precision inference
        conf_thres = 0.3  # confidence threshold
        iou_thres = 0.05  # NMS IOU threshold

        max_det = 1000  # maximum detections per image
        imgsz = [640, 640]
        names = {
            0: 'photovoltaic_panels',
            1: 'stripe_spot',
            2: 'single_spot',
            3: 'multi_spot'
        }

        stride = 32
        fake_result = {}
        fake_result["model_data"] = {"objects": []}

        img = letterbox(input_image, imgsz, stride, True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img / 255  # 0 - 255 to 0.0 - 1.0
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, dim=0).to(device)
        img = img.type(torch.cuda.HalfTensor)
        pred = handle(img, augment=False)[0]

        fake_result = {}
        n = 0
        algorithm_list = []
        fake_result["algorithm_data"] = {
            "is_alert": False,
            "target_count": n,
            "target_info": []
        }
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False)
        for i, det in enumerate(pred):  # per image
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], input_image.shape).round()

            for *xyxy, conf, cls in reversed(det):
                xyxy_list = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                conf_list = conf.tolist()
                label = names[int(cls)]
                fake_result['model_data']['objects'].append({
                    "xmin": int(xyxy_list[0]),
                    "ymin": int(xyxy_list[1]),
                    "xmax": int(xyxy_list[2]),
                    "ymax": int(xyxy_list[3]),
                    "confidence": conf_list,
                    "name": label
                })
                if label == 'stripe_spot' or label == 'single_spot' or label == 'multi_spot':
                    n += 1
                    algorithm_list.append({
                        "xmin": int(xyxy_list[0]),
                        "ymin": int(xyxy_list[1]),
                        "xmax": int(xyxy_list[2]),
                        "ymax": int(xyxy_list[3]),
                        "confidence": conf_list,
                        "name": label
                    })
                    fake_result["algorithm_data"].append({
                        "is_alert": True,
                        "target_count": n,
                        "target_info": algorithm_list
                    })

    return json.dumps(fake_result, indent=4)


if __name__ == '__main__':
    # Test API
    img = cv2.imread(r'/home/data/1216/SXphotovoltaic20220704_V1_train_factory_out_1_002384.jpg')
    predictor = init()
    import time

    s = time.time()
    fake_result = process_image(predictor, img)
    e = time.time()
    print(fake_result)
    print((e - s))
