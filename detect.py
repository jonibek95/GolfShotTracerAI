

# import argparse
# import csv
# import os
# import platform
# import sys
# from pathlib import Path
# import numpy as np
# import cv2

# import torch
# # from Bezier_Curve.Bezier import CubicBezier 
# from my_functions import BezierCurve
# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from ultralytics.utils.plotting import Annotator, colors, save_one_box

# from models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import (
#     LOGGER,
#     Profile,
#     check_file,
#     check_img_size,
#     check_imshow,
#     check_requirements,
#     colorstr,
#     cv2,
#     increment_path,
#     non_max_suppression,
#     print_args,
#     scale_boxes,
#     strip_optimizer,
#     xyxy2xywh,
# )
# from utils.torch_utils import select_device, smart_inference_mode


# @smart_inference_mode()
# def run(
#     weights=ROOT / "yolov5s.pt",  # model path or triton URL
#     source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
#     data=ROOT / "data/Objects365.yaml",  # dataset.yaml path
#     imgsz=(640, 640),  # inference size (height, width)
#     conf_thres=0.25,  # confidence threshold
#     iou_thres=0.45,  # NMS IOU threshold
#     max_det=1000,  # maximum detections per image
#     device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
#     view_img=False,  # show results
#     save_txt=False,  # save results to *.txt
#     save_csv=False,  # save results in CSV format
#     save_conf=False,  # save confidences in --save-txt labels
#     save_crop=False,  # save cropped prediction boxes
#     nosave=False,  # do not save images/videos
#     classes=None,  # filter by class: --class 0, or --class 0 2 3
#     agnostic_nms=False,  # class-agnostic NMS
#     augment=False,  # augmented inference
#     visualize=False,  # visualize features
#     update=False,  # update all models
#     project=ROOT / "runs/detect",  # save results to project/name
#     name="exp",  # save results to project/name
#     exist_ok=False,  # existing project/name ok, do not increment
#     line_thickness=3,  # bounding box thickness (pixels)
#     hide_labels=False,  # hide labels
#     hide_conf=False,  # hide confidences
#     half=False,  # use FP16 half-precision inference
#     dnn=False,  # use OpenCV DNN for ONNX inference
#     vid_stride=1,  # video frame-rate stride
# ):
#     source = str(source)
#     save_img = not nosave and not source.endswith(".txt")  # save inference images
#     is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
#     is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
#     webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
#     screenshot = source.lower().startswith("screen")
#     if is_url and is_file:
#         source = check_file(source)  # download

#     # Directories
#     save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
#     (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

#     # Load model
#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
#     stride, names, pt = model.stride, model.names, model.pt
#     imgsz = check_img_size(imgsz, s=stride)  # check image size

#     # Dataloader
#     bs = 1  # batch_size
#     if webcam:
#         view_img = check_imshow(warn=True)
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#         bs = len(dataset)
#     elif screenshot:
#         dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#     vid_path, vid_writer = [None] * bs, [None] * bs

#     # Run inference
#     model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
#     seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    
#     centroids = []  # Store centroids of detected golf balls
#     frame_count = 0  # Frame counter
#     max_frames = 30  # Number of frames to collect centroids
#     predict_trajectory = False  # Flag to start prediction
#     launch_angle = 45  # Launch angle in degrees
#     apex_height = 50  # APEX height in meters
#     predicted_trajectory = []
#     detected_positions = []  # This will store tuples of (frame_idx, (x, y))
#     kf = setup_kalman_filter()  # Assuming you have a Kalman Filter setup function  
    
#     for path, im, im0s, vid_cap, s in dataset:
#         frame_count += 1
#         with dt[0]:
#             im = torch.from_numpy(im).to(model.device)
#             im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#             im /= 255  # 0 - 255 to 0.0 - 1.0
#             if len(im.shape) == 3:
#                 im = im[None]  # expand for batch dim
#             if model.xml and im.shape[0] > 1:
#                 ims = torch.chunk(im, im.shape[0], 0)

#         # Inference
#         with dt[1]:
#             visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
#             if model.xml and im.shape[0] > 1:
#                 pred = None
#                 for image in ims:
#                     if pred is None:
#                         pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
#                     else:
#                         pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
#                 pred = [pred, None]
#             else:
#                 pred = model(im, augment=augment, visualize=visualize)
#         # NMS
#         with dt[2]:
#             pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


#         # Define the path for the CSV file
#         csv_path = save_dir / "predictions.csv"

#         # Create or append to the CSV file
#         def write_to_csv(image_name, prediction, confidence):
#             """Writes prediction data for an image to a CSV file, appending if the file exists."""
#             data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
#             with open(csv_path, mode="a", newline="") as f:
#                 writer = csv.DictWriter(f, fieldnames=data.keys())
#                 if not csv_path.is_file():
#                     writer.writeheader()
#                 writer.writerow(data)

#         centers = []  # List to store centers of boxes
#         # Process predictions
#         for i, det in enumerate(pred):  # per image
#             seen += 1
#             if webcam:  # batch_size >= 1
#                 p, im0, frame = path[i], im0s[i].copy(), dataset.count
#                 s += f"{i}: "
#             else:
#                 p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

#             p = Path(p)  # to Path
#             save_path = str(save_dir / p.name)  # im.jpg
#             txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
#             s += "%gx%g " % im.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             imc = im0.copy() if save_crop else im0  # for save_crop
#             annotator = Annotator(im0, line_width=line_thickness, example=str(names))
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

#                 # Print results
#                 for c in det[:, 5].unique():
#                     n = (det[:, 5] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                
                
#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     c = int(cls)  # integer class
#                     label = names[c] if hide_conf else f"{names[c]}"
#                     confidence = float(conf)
#                     confidence_str = f"{confidence:.2f}"
                    
#                     # x1, y1, x2, y2 = map(int, xyxy)
#                     # center = ((x1 + x2) // 2, (y1 + y2) // 2)
#                     # centers.append(center)
#                     # if frame_count <= max_frames:
#                     #     centroids.append(center)
#                     x, y = (int(xyxy[0] + xyxy[2]) // 2, int(xyxy[1] + xyxy[3]) // 2)
#                     detected_positions.append((frame_count, (x, y)))
#                     kf.update(np.array([[x], [y]]))  # Update the filter with the new detection
                    
                    

#                     if save_csv:
#                         write_to_csv(p.name, label, confidence_str)

#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#                         with open(f"{txt_path}.txt", "a") as f:
#                             f.write(("%g " * len(line)).rstrip() % line + "\n")

#                     if save_img or save_crop or view_img:  # Add bbox to image
#                         c = int(cls)  # integer class
#                         label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
#                         annotator.box_label(xyxy, color=colors(c, True))
#                     if save_crop:
#                         save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
                        
            
                    
                    
            
#             # if len(center_points) > 1:
#             #     t_values = np.linspace(0, 1, num=len(center_points))
#             #     bezier_curve = BezierCurve.get_curved_path(center_points, t_values)
#             #     bezier_points = np.array(bezier_curve.points, dtype=np.int32).reshape((-1, 1, 2))
#             #     cv2.polylines(im0, [bezier_points], False, (0, 255, 0), 2)
#             # Predict and draw trajectory for every frame
#             kf.predict()
#             predicted_x, predicted_y = int(kf.x[0]), int(kf.x[1])
#             detected_positions.append((frame_count, (predicted_x, predicted_y)))
            
#             # Ensure only positions are passed to draw_trajectory
#             if len(detected_positions) > 1:
#                 trajectory_positions = [pos[1] for pos in detected_positions]  # Extract only coordinate tuples
#                 draw_trajectory(im0, trajectory_positions)
            
            
#             im0 = annotator.result()
#             if view_img:
#                 if platform.system() == "Linux" and p not in windows:
#                     windows.append(p)
#                     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
#                     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
#                 cv2.imshow(str(p), im0)
#                 cv2.waitKey(1)  # 1 millisecond

#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == "image":
#                     cv2.imwrite(save_path, im0)
#                 else:  # 'video' or 'stream'
#                     if vid_path[i] != save_path:  # new video
#                         vid_path[i] = save_path
#                         if isinstance(vid_writer[i], cv2.VideoWriter):
#                             vid_writer[i].release()  # release previous video writer
#                         if vid_cap:  # video
#                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:  # stream
#                             fps, w, h = 30, im0.shape[1], im0.shape[0]
#                         save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
#                         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
#                     vid_writer[i].write(im0)
    
#             # Print time (inference-only)
#             LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

#     # Print results
#     t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
#     LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
#     if save_txt or save_img:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
#         LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
#     if update:
#         strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
        
# def setup_kalman_filter():
#     kf = KalmanFilter(dim_x=4, dim_z=2)
#     dt = 1  # Time step
#     kf.F = np.array([[1, 0, dt, 0],  # State transition matrix
#                      [0, 1, 0, dt],
#                      [0, 0, 1,  0],
#                      [0, 0, 0,  1]])
#     kf.H = np.array([[1, 0, 0, 0],
#                      [0, 1, 0, 0]])
#     kf.R *= np.array([[100, 0],    # Measurement noise matrix
#                       [0, 100]])
#     kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.01)  # Process noise
#     kf.P *= 1000  # Initial uncertainty
#     return kf

# def draw_trajectory(im0, positions):
#     if len(positions) > 1:
#         for i in range(1, len(positions)):
#             prev_x, prev_y = positions[i - 1]
#             current_x, current_y = positions[i]
#             cv2.line(im0, (prev_x, prev_y), (current_x, current_y), (0, 255, 0), 2)


# def parse_opt():
#     """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
#     parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
#     parser.add_argument("--data", type=str, default=ROOT / "data/Objects365.yaml", help="(optional) dataset.yaml path")
#     parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
#     parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
#     parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
#     parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
#     parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
#     parser.add_argument("--view-img", action="store_true", help="show results")
#     parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
#     parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
#     parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
#     parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
#     parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
#     parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
#     parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
#     parser.add_argument("--augment", action="store_true", help="augmented inference")
#     parser.add_argument("--visualize", action="store_true", help="visualize features")
#     parser.add_argument("--update", action="store_true", help="update all models")
#     parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
#     parser.add_argument("--name", default="exp", help="save results to project/name")
#     parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
#     parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
#     parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
#     parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
#     parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
#     parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
#     parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#     print_args(vars(opt))
#     return opt


# def main(opt):
#     """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
#     check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
#     run(**vars(opt))


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)






















import os
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
from argparse import ArgumentParser

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class YOLODetector:
    def __init__(self,
        weights = None,  # model.pt path(x)
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        imgsz=[640, 640],  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,
        save_txt=False,
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,
        dnn=False,  # use OpenCV DNN for ONNX inference
        save_conf=False,
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        nosave=False,  # do not save images/videos
        save_crop = False,
        update=False,
        ):
        self.weights = weights
        self.imgsz = imgsz
        self.max_det = max_det
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.dnn = dnn
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.nosave = nosave
        self.update = update
        
        # Load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.dt, self.seen = [0.0, 0.0, 0.0], 0
    
    def Prediction(self, image):
        dataset = LoadImages(image, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        highest_conf = 0
        highest_conf_box = None
        all_boxes = []

        for path, im, im0s, _, _ in dataset:
            im = torch.from_numpy(im).to(self.device).float() / 255.0  # Normalize and add batch dimension
            if len(im.shape) == 3:
                im = im.unsqueeze(0)

            pred = self.model(im, augment=self.augment, visualize=self.visualize)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            for det in pred:  # Process detections
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()  # Scale boxes
                    for *xyxy, conf, cls in reversed(det):
                        if conf > highest_conf:
                            highest_conf = conf
                            x1, y1, x2, y2 = [int(x.item()) for x in xyxy]
                            highest_conf_box = [x1, y1, x2, y2]  # Update highest confidence box
            if highest_conf_box:
                all_boxes.append(highest_conf_box) 

        return all_boxes   # Return the last processed frame and highest confidence box

