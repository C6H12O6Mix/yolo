import cv2
import time
import logging
import numpy as np
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RTMP-Stream")

class RTMPStream:
    def __init__(
        self, 
        rtmp_url: str,
        buffer_size: int = 10,
        reconnect_attempts: int = 5,
        reconnect_delay: int = 3
    ):
        """
        初始化RTMP流处理器
        
        参数:
            rtmp_url: RTMP流地址
            buffer_size: 帧缓冲区大小
            reconnect_attempts: 重连尝试次数
            reconnect_delay: 重连延迟(秒)
        """
        self.rtmp_url = rtmp_url
        self.buffer_size = buffer_size
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.frame_buffer = []
        self.is_running = False
        
    def connect(self) -> bool:
        """连接到RTMP流"""
        logger.info(f"正在连接到RTMP流: {self.rtmp_url}")
        
        for attempt in range(self.reconnect_attempts):
            try:
                self.cap = cv2.VideoCapture(self.rtmp_url)
                if self.cap.isOpened():
                    logger.info("RTMP流连接成功")
                    return True
                else:
                    logger.warning(f"无法打开RTMP流，尝试 {attempt+1}/{self.reconnect_attempts}")
            except Exception as e:
                logger.error(f"连接RTMP流时出错: {str(e)}")
            
            time.sleep(self.reconnect_delay)
        
        logger.error(f"在 {self.reconnect_attempts} 次尝试后无法连接到RTMP流")
        return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧，处理可能的连接问题"""
        if not self.cap or not self.cap.isOpened():
            if not self.connect():
                return None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("读取帧失败，尝试重新连接")
                self.cap.release()
                self.cap = None
                if self.connect():
                    return self.read_frame()
                return None
            return frame
        except Exception as e:
            logger.error(f"读取帧时出错: {str(e)}")
            self.cap.release()
            self.cap = None
            return None
    
    def start(self):
        """开始处理RTMP流"""
        if not self.connect():
            return
        
        self.is_running = True
        logger.info("开始处理RTMP流")
        
        try:
            while self.is_running:
                frame = self.read_frame()
                if frame is not None:
                    # 管理帧缓冲区
                    if len(self.frame_buffer) >= self.buffer_size:
                        self.frame_buffer.pop(0)
                    self.frame_buffer.append(frame)
                    
                    # 这里可以添加帧处理逻辑
                    # process_frame(frame)
                else:
                    time.sleep(0.1)  # 避免CPU占用过高
        except KeyboardInterrupt:
            logger.info("用户中断流处理")
        finally:
            self.stop()
    
    def stop(self):
        """停止处理RTMP流"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("RTMP流处理已停止")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新的帧"""
        if self.frame_buffer:
            return self.frame_buffer[-1]
        return None

# 使用示例
if __name__ == "__main__":
    rtmp_url = "rtmp://localhost:1935/live/stream"
    stream = RTMPStream(rtmp_url)
    stream.start() 