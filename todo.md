# YOLO视频流处理项目精简计划

## 核心功能 - 视频流处理与OBB分析
- [x] 配置RTMP服务器
  - [x] 安装Nginx并配置RTMP模块
  - [x] 设置服务器监听端口和应用名称
  - [x] 配置第三方推流接收端点

- [x] 实现YOLO拉流功能
  - [x] 配置YOLO内置拉流组件
  - [x] 设置拉流参数和缓冲策略
  - [x] 处理流连接异常情况

- [x] 部署YOLOv11-OBB模型
  - [x] 下载预训练权重文件
  - [x] 转换模型为推理格式
  - [x] 配置模型参数和阈值

- [x] 开发OBB目标检测功能
  - [x] 实现预处理图像函数
  - [x] 开发模型推理接口
  - [x] 添加后处理结果解析

- [x] 实现检测结果可视化
  - [x] 开发OBB旋转框绘制函数
  - [x] 实现类别颜色映射
  - [x] 添加置信度显示

- [x] 配置处理后视频回传
  - [x] 实现帧合成视频流
  - [x] 配置ffmpeg编码参数
  - [x] 开发RTMP推流接口

- [x] 测试端到端视频处理流程
  - [x] 测量端到端延迟
  - [x] 评估视频质量
  - [x] 分析资源占用情况

## Docker容器化
- [x] 创建Dockerfile
  - [x] 选择合适的基础镜像(如Python+CUDA)
  - [x] 安装系统依赖(ffmpeg, nginx-rtmp)
  - [x] 配置Python环境和依赖
  - [x] 复制应用代码和模型

- [x] 配置容器环境
  - [x] 设置环境变量和配置文件
  - [x] 配置端口映射(RTMP端口)
  - [x] 设置持久化存储卷
  - [x] 优化容器资源分配

- [x] 创建Docker Compose配置
  - [x] 定义服务组件
  - [x] 配置服务间通信
  - [x] 设置自动重启策略
  - [x] 添加健康检查

- [x] 测试Docker部署
  - [x] 验证容器启动和服务可用性
  - [x] 测试RTMP推流和拉流功能
  - [x] 检查资源使用情况
  - [x] 验证服务持久性和稳定性

## 技术栈
- 视频流处理: RTMP, ffmpeg
  - nginx-rtmp-module
  - FFmpeg 5.0+
- 目标检测: YOLOv11-OBB
  - PyTorch 2.0+
  - ONNX Runtime (可选)
- 图像处理: OpenCV
  - OpenCV 4.5+
- 后端API: 简化版FastAPI
  - Uvicorn ASGI服务器
- 容器化: Docker
  - Docker Compose
  - NVIDIA Container Toolkit (GPU支持) 