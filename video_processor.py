import cv2
import time
import subprocess
import threading
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple
from rtmp_stream import RTMPStream
from yolo_obb_model import YOLOv11OBB

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Video-Processor")

class VideoProcessor:
    def __init__(
        self, 
        input_rtmp_url: str,
        output_rtmp_url: str,
        model_weights_path: str,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        bitrate: str = "2000k"
    ):
        """
        初始化视频处理器
        
        参数:
            input_rtmp_url: 输入RTMP流URL
            output_rtmp_url: 输出RTMP流URL
            model_weights_path: YOLO模型权重路径
            fps: 输出视频帧率
            width: 输出视频宽度
            height: 输出视频高度
            bitrate: 输出视频比特率
        """
        self.input_rtmp_url = input_rtmp_url
        self.output_rtmp_url = output_rtmp_url
        self.model_weights_path = model_weights_path
        self.fps = fps
        self.width = width
        self.height = height
        self.bitrate = bitrate
        
        # 初始化RTMP流读取器
        self.stream_reader = RTMPStream(input_rtmp_url)
        
        # 初始化YOLO-OBB模型
        self.model = YOLOv11OBB(model_weights_path)
        
        # 初始化FFmpeg进程
        self.ffmpeg_process = None
        
        # 处理状态
        self.is_processing = False
        self.processing_thread = None
        
        # 性能指标
        self.metrics = {
            'fps': 0,
            'latency': 0,
            'detection_time': 0,
            'processing_time': 0
        }
    
    def _init_ffmpeg(self):
        """初始化FFmpeg进程用于RTMP推流"""
        try:
            # FFmpeg命令
            command = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pixel_format', 'bgr24',
                '-video_size', f"{self.width}x{self.height}",
                '-framerate', str(self.fps),
                '-i', '-',  # 从stdin读取
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-b:v', self.bitrate,
                '-maxrate', self.bitrate,
                '-bufsize', self.bitrate,
                '-f', 'flv',
                self.output_rtmp_url
            ]
            
            logger.info(f"启动FFmpeg推流: {' '.join(command)}")
            
            # 创建FFmpeg进程
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            
            logger.info("FFmpeg进程已启动")
            return True
        except Exception as e:
            logger.error(f"初始化FFmpeg时出错: {str(e)}")
            return False
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧图像
        
        参数:
            frame: 输入帧
            
        返回:
            处理后的帧
        """
        if frame is None:
            return None
        
        # 调整帧大小
        frame = cv2.resize(frame, (self.width, self.height))
        
        # 记录开始时间
        start_time = time.time()
        
        # 运行目标检测
        detections = self.model.detect(frame)
        
        # 记录检测时间
        detection_time = time.time() - start_time
        self.metrics['detection_time'] = detection_time
        
        # 绘制检测结果
        processed_frame = self.model.draw_detections(frame, detections)
        
        # 添加性能指标
        self._draw_metrics(processed_frame)
        
        # 记录总处理时间
        processing_time = time.time() - start_time
        self.metrics['processing_time'] = processing_time
        
        return processed_frame
    
    def _draw_metrics(self, frame: np.ndarray):
        """在帧上绘制性能指标"""
        # 帧率
        fps_text = f"FPS: {self.metrics['fps']:.1f}"
        cv2.putText(
            frame, fps_text, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # 延迟
        latency_text = f"Latency: {self.metrics['latency']:.1f} ms"
        cv2.putText(
            frame, latency_text, (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # 检测时间
        det_time_text = f"Det Time: {self.metrics['detection_time']*1000:.1f} ms"
        cv2.putText(
            frame, det_time_text, (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    
    def _processing_loop(self):
        """视频处理主循环"""
        # 初始化FFmpeg
        if not self._init_ffmpeg():
            logger.error("无法启动处理循环，FFmpeg初始化失败")
            return
        
        # 帧计数和FPS计算
        frame_count = 0
        fps_start_time = time.time()
        
        logger.info("开始视频处理循环")
        
        while self.is_processing:
            # 获取开始时间用于计算延迟
            start_time = time.time()
            
            # 获取最新帧
            frame = self.stream_reader.get_latest_frame()
            
            if frame is not None:
                # 处理帧
                processed_frame = self._process_frame(frame)
                
                if processed_frame is not None:
                    # 计算延迟
                    latency = (time.time() - start_time) * 1000  # 毫秒
                    self.metrics['latency'] = latency
                    
                    # 将处理后的帧写入FFmpeg进程
                    try:
                        self.ffmpeg_process.stdin.write(processed_frame.tobytes())
                    except BrokenPipeError:
                        logger.error("FFmpeg进程管道已断开")
                        break
                    except Exception as e:
                        logger.error(f"写入FFmpeg进程时出错: {str(e)}")
                        break
                    
                    # 更新帧计数
                    frame_count += 1
                    
                    # 每秒计算一次FPS
                    if frame_count % 30 == 0:
                        current_time = time.time()
                        elapsed = current_time - fps_start_time
                        if elapsed > 0:
                            self.metrics['fps'] = frame_count / elapsed
                            frame_count = 0
                            fps_start_time = current_time
            else:
                # 如果没有帧，短暂休眠以减少CPU使用
                time.sleep(0.01)
        
        # 关闭FFmpeg进程
        if self.ffmpeg_process:
            logger.info("关闭FFmpeg进程")
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait(timeout=5)
            except Exception as e:
                logger.error(f"关闭FFmpeg进程时出错: {str(e)}")
                self.ffmpeg_process.kill()
            
            self.ffmpeg_process = None
    
    def start(self):
        """启动视频处理"""
        if self.is_processing:
            logger.warning("视频处理已经在运行")
            return
        
        logger.info("启动视频处理")
        
        # 启动RTMP流读取
        self.stream_reader.start()
        
        # 设置处理状态
        self.is_processing = True
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("视频处理已启动")
    
    def stop(self):
        """停止视频处理"""
        if not self.is_processing:
            logger.warning("视频处理未运行")
            return
        
        logger.info("停止视频处理")
        
        # 设置处理状态
        self.is_processing = False
        
        # 等待处理线程结束
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            self.processing_thread = None
        
        # 停止RTMP流读取
        self.stream_reader.stop()
        
        logger.info("视频处理已停止")
    
    def get_metrics(self) -> Dict:
        """获取性能指标"""
        return self.metrics

# 使用示例
if __name__ == "__main__":
    # 配置参数
    input_rtmp_url = "rtmp://localhost:1935/live/stream"
    output_rtmp_url = "rtmp://localhost:1935/live/processed"
    model_weights_path = "weights/yolo11n-obb.pt"
    
    # 创建视频处理器
    processor = VideoProcessor(
        input_rtmp_url=input_rtmp_url,
        output_rtmp_url=output_rtmp_url,
        model_weights_path=model_weights_path
    )
    
    # 启动处理
    processor.start()
    
    try:
        # 运行一段时间
        while True:
            time.sleep(1)
            metrics = processor.get_metrics()
            print(f"FPS: {metrics['fps']:.1f}, Latency: {metrics['latency']:.1f} ms")
    except KeyboardInterrupt:
        print("用户中断，停止处理")
    finally:
        # 停止处理
        processor.stop() 