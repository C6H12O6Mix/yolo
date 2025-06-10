# YOLO视频流处理项目

这是一个基于YOLOv11-OBB的视频流处理项目，用于实时检测和分析视频流中的目标。

## 主要功能
- RTMP视频流接收与处理
- YOLOv11-OBB目标检测与分析
- 检测结果可视化
- 处理后视频回传

## 技术栈
- 视频流处理: RTMP, ffmpeg
- 目标检测: YOLOv11-OBB
- 图像处理: OpenCV
- 后端API: FastAPI
- 容器化: Docker

## 项目结构
```
├── api_server.py       # FastAPI服务器
├── docker-compose.yml  # Docker Compose配置
├── Dockerfile          # Docker镜像构建文件
├── main.py             # 主程序入口
├── nginx.conf          # Nginx RTMP服务器配置
├── requirements.txt    # Python依赖项
├── rtmp_stream.py      # RTMP流处理模块
├── start_docker.bat    # Windows Docker启动脚本
├── start_windows.bat   # Windows本地启动脚本
├── test_pipeline.py    # 端到端测试脚本
├── todo.md             # 项目任务清单
├── video_processor.py  # 视频处理模块
├── weights/            # 模型权重目录
└── yolo_obb_model.py   # YOLOv11-OBB模型实现
```

## 环境准备

### 依赖项
- Python 3.8+
- CUDA 11.0+ (GPU加速，可选)
- FFmpeg 5.0+
- Nginx with RTMP module

### 安装依赖
```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装系统依赖 (Ubuntu)
sudo apt-get update
sudo apt-get install -y ffmpeg nginx libnginx-mod-rtmp
```

## 模型权重准备

本项目使用YOLO11n-OBB模型，这是一个轻量级模型，具有以下特点：
- mAP50: 78.4
- 速度: 在CPU上117.6ms，在T4 GPU上仅需4.4ms
- 参数量: 2.7M
- 计算量: 17.2B FLOPs

### 下载模型权重
1. 创建weights目录（如果不存在）
2. 下载YOLO11n-OBB预训练权重文件并放入weights目录：
```bash
mkdir -p weights
# 下载权重文件 (示例链接，请替换为实际链接)
wget -O weights/yolo11n-obb.pt https://example.com/yolo11n-obb.pt
```

## Windows用户快速开始

### 方法1: 使用启动脚本（推荐）
1. 确保已安装Python和所需依赖
2. 下载YOLO11n-OBB模型权重文件到weights目录
3. 双击运行`start_windows.bat`

### 方法2: 使用Docker（最简单）
1. 安装Docker Desktop并启动
2. 下载YOLO11n-OBB模型权重文件到weights目录
3. 双击运行`start_docker.bat`

## 本地运行

### 1. 准备模型权重
确保已将YOLO11n-OBB模型权重文件放入weights目录

### 2. 启动Nginx RTMP服务器（Linux/Mac）
```bash
nginx -c $(pwd)/nginx.conf
```

### 3. 启动API服务器
```bash
python api_server.py
```

### 4. 或使用主程序一键启动
```bash
# Linux/Mac
python main.py

# Windows (跳过Nginx)
python main.py --skip-nginx
```

## Docker部署

### 1. 构建Docker镜像
```bash
# 确保模型权重文件已放入weights目录
docker build -t yolo-video-processor .
```

### 2. 使用Docker Compose启动服务
```bash
docker-compose up -d
```
注意：Docker配置已优化，可以直接使用本地weights目录中的模型文件，无需重新构建镜像。

### 3. 查看服务日志
```bash
docker-compose logs -f
```

### 4. 停止服务
```bash
docker-compose down
```

## 测试流程

### 1. 准备测试视频
准备一个测试视频文件，例如test.mp4

### 2. 使用测试脚本进行端到端测试
```bash
# 本地测试
python test_pipeline.py --video path/to/test.mp4 --duration 60

# Docker环境测试
python test_pipeline.py --video path/to/test.mp4 --rtmp-server localhost --api-url http://localhost:8000 --duration 60
```

### 3. 查看测试结果
测试脚本会输出性能指标，包括：
- FPS (每秒处理帧数)
- 延迟 (端到端处理延迟)
- 检测时间 (模型推理时间)

### 4. 手动测试
也可以使用FFmpeg手动推流：
```bash
ffmpeg -re -i test.mp4 -c:v libx264 -preset veryfast -tune zerolatency -c:a aac -ar 44100 -f flv rtmp://localhost:1935/live/input
```

然后通过API启动处理：
```bash
curl -X POST http://localhost:8000/start -H "Content-Type: application/json" -d '{
  "input_rtmp_url": "rtmp://localhost:1935/live/input",
  "output_rtmp_url": "rtmp://localhost:1935/live/output",
  "model_weights_path": "weights/yolo11n-obb.pt",
  "fps": 30,
  "width": 1280,
  "height": 720
}'
```

使用VLC或FFplay查看处理后的视频流：
```bash
ffplay rtmp://localhost:1935/live/output
```

## 性能优化
- 调整模型配置参数（conf_threshold, iou_threshold）以平衡精度和速度
- 使用较小的输入分辨率提高处理速度
- 在资源充足的GPU环境下运行以获得最佳性能
