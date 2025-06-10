import os
import cv2
import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("YOLO-OBB")

class YOLOv11OBB:
    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: str = None
    ):
        """
        初始化YOLOv11-OBB模型
        
        参数:
            weights_path: 模型权重文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            img_size: 输入图像大小
            device: 运行设备 ('cpu', 'cuda', 'cuda:0', 等)
        """
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        # 如果未指定设备，自动检测
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载模型
        self._load_model()
        
        # 加载类别名称
        self.class_names = self._load_class_names()
        
        # 生成颜色映射
        self.colors = self._generate_colors()
        
    def _load_model(self):
        """加载YOLOv11-OBB模型"""
        try:
            logger.info(f"正在加载YOLOv11-OBB模型: {self.weights_path}")
            # 检查权重文件是否存在
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"模型权重文件不存在: {self.weights_path}")
            
            # 使用PyTorch加载模型
            # 注意: 这里使用的是伪代码，实际实现需要根据YOLOv11-OBB的具体API
            self.model = torch.hub.load('ultralytics/yolov11-obb', 'custom', path=self.weights_path)
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"模型加载成功，运行设备: {self.device}")
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise
    
    def _load_class_names(self) -> List[str]:
        """加载类别名称"""
        # 这里应该从配置文件或模型中加载实际的类别名称
        # 以下是示例类别
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]
    
    def _generate_colors(self) -> List[Tuple[int, int, int]]:
        """为每个类别生成唯一的颜色"""
        np.random.seed(42)  # 固定随机种子以保持颜色一致
        return [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 
                for _ in range(len(self.class_names))]
    
    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        """预处理图像用于模型输入"""
        # 调整图像大小
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 转换为RGB (OpenCV默认是BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化并转换为张量
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).to(self.device)
        
        return img
    
    def detect(self, img: np.ndarray) -> List[Dict]:
        """
        在图像上进行OBB目标检测
        
        参数:
            img: 输入图像 (BGR格式)
            
        返回:
            检测结果列表，每个结果包含:
            - box: 旋转框坐标 [cx, cy, w, h, angle]
            - conf: 置信度
            - cls_id: 类别ID
            - cls_name: 类别名称
        """
        try:
            # 保存原始图像尺寸用于后处理
            orig_h, orig_w = img.shape[:2]
            
            # 预处理图像
            processed_img = self.preprocess(img)
            
            # 模型推理
            with torch.no_grad():
                results = self.model(processed_img)
            
            # 解析结果 (伪代码，根据实际YOLOv11-OBB API调整)
            detections = []
            
            # 假设results包含检测到的旋转框、置信度和类别
            for det in results.obb:  # 假设results有obb属性
                for *box, conf, cls_id in det:
                    if conf >= self.conf_threshold:
                        # 将坐标转换回原始图像尺寸
                        cx, cy, w, h, angle = box
                        cx *= orig_w / self.img_size
                        cy *= orig_h / self.img_size
                        w *= orig_w / self.img_size
                        h *= orig_h / self.img_size
                        
                        cls_id = int(cls_id)
                        cls_name = self.class_names[cls_id]
                        
                        detections.append({
                            'box': [cx, cy, w, h, angle],
                            'conf': float(conf),
                            'cls_id': cls_id,
                            'cls_name': cls_name
                        })
            
            return detections
        except Exception as e:
            logger.error(f"检测过程中出错: {str(e)}")
            return []
    
    def draw_detections(self, img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        参数:
            img: 原始图像
            detections: 检测结果列表
            
        返回:
            带有检测结果的图像
        """
        result_img = img.copy()
        
        for det in detections:
            cx, cy, w, h, angle = det['box']
            conf = det['conf']
            cls_id = det['cls_id']
            cls_name = det['cls_name']
            
            # 获取类别对应的颜色
            color = self.colors[cls_id]
            
            # 计算旋转框的四个角点
            rect = ((cx, cy), (w, h), angle * 180.0 / np.pi)  # OpenCV使用角度制
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 绘制旋转框
            cv2.drawContours(result_img, [box], 0, color, 2)
            
            # 绘制类别名称和置信度
            label = f"{cls_name} {conf:.2f}"
            font_scale = 0.6
            font_thickness = 1
            
            # 获取文本大小
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # 计算文本框位置
            text_offset_x = int(cx - text_width / 2)
            text_offset_y = int(cy - h / 2 - 5)
            
            # 确保文本框在图像内
            text_offset_x = max(text_offset_x, 0)
            text_offset_y = max(text_offset_y, text_height)
            
            # 绘制文本背景
            cv2.rectangle(
                result_img,
                (text_offset_x, text_offset_y - text_height),
                (text_offset_x + text_width, text_offset_y),
                color,
                -1
            )
            
            # 绘制文本
            cv2.putText(
                result_img,
                label,
                (text_offset_x, text_offset_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        return result_img

# 使用示例
if __name__ == "__main__":
    # 模型权重路径
    weights_path = "weights/yolo11n-obb.pt"
    
    # 初始化模型
    model = YOLOv11OBB(weights_path)
    
    # 加载测试图像
    img = cv2.imread("test.jpg")
    
    # 运行检测
    detections = model.detect(img)
    
    # 绘制结果
    result_img = model.draw_detections(img, detections)
    
    # 显示结果
    cv2.imshow("YOLOv11-OBB Detection", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 