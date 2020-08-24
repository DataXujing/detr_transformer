
'''
detr 单张图片的推断
'''
import cv2
from PIL import Image
import numpy as np
import os

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 当前使用{}做推断".format(device))


# 图像数据处理
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# 将0-1映射到图像
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b




# plot box by opencv
# plot box by opencv
def plot_result(pil_img, prob, boxes,save_name=None,imshow=False, imwrite=False):
    LABEL = ["N/A","QP","NY","QG"]
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # print(prob)

    # print("-------------------------------")

    # print(boxes)

    if len(prob) == 0:
        print("[INFO] NO box detect !!! ")
        if imwrite:
            if not os.path.exists("./result/pred_no"):
                os.makedirs("./result/pred_no")
            cv2.imwrite(os.path.join("./result/pred_no",save_name),opencvImage)
        return opencvImage

    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):

        cl = p.argmax()
        label_text = '{}: {}%'.format(LABEL[cl],round(p[cl]*100,2))

        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
        cv2.putText(opencvImage, label_text,(int(xmin)+10, int(ymin)+30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 255, 0), 2)

    if imshow:
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)

    if imwrite:
        if not os.path.exists("./result/pred"):
            os.makedirs('./result/pred')
        cv2.imwrite('./result/pred/{}'.format(save_name), opencvImage)

    return opencvImage




# 单张图像的推断
def detect(im, model, transform,prob_threshold=0.7):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > prob_threshold

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled



if __name__ == "__main__":
    detr = detr_resnet50(pretrained=False,num_classes=3).eval()  # <------这里类别不需要+1
    state_dict =  torch.load('outputs/checkpoint299.pth')   # <-----------修改加载模型的路径
    detr.load_state_dict(state_dict["model"])
    detr.to(device)

    # video
    vid = cv2.VideoCapture("./test.mp4")
    video_frame_cnt = int(vid.get(7))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    video_fps = int(vid.get(5))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('output.mp4', fourcc, video_fps, (video_width, video_height))

    for i in range(video_frame_cnt):
        print("{}/{}".format(i,video_frame_cnt))
        ret, img_ori = vid.read()
        image_pil = Image.fromarray(cv2.cvtColor(img_ori,cv2.COLOR_BGR2RGB))  

        scores, boxes = detect(image_pil, detr, transform)
        image = plot_result(image_pil, scores, boxes,save_name=file,imshow=False, imwrite=True)

        videoWriter.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    vid.release()
    videoWriter.release()



