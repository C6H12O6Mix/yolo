import cv2
import time
import requests
import subprocess
import os
import argparse
import logging
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Test-Pipeline")

class TestPipeline:
    def __init__(
        self,
        test_video_path: str,
        rtmp_server: str = "localhost",
        rtmp_port: int = 1935,
        api_url: str = "http://localhost:8000",
        model_path: str = "weights/yolo11n-obb.pt"
    ):
        """
        初始化测试管道
        
        参数:
            test_video_path: 测试视频路径
            rtmp_server: RTMP服务器地址
            rtmp_port: RTMP服务器端口
            api_url: API服务器URL
            model_path: 模型路径
        """
        self.test_video_path = test_video_path
        self.rtmp_server = rtmp_server
        self.rtmp_port = rtmp_port
        self.api_url = api_url
        self.model_path = model_path
        
        # RTMP URL
        self.input_rtmp_url = f"rtmp://{rtmp_server}:{rtmp_port}/live/input"
        self.output_rtmp_url = f"rtmp://{rtmp_server}:{rtmp_port}/live/output"
        
        # FFmpeg进程
        self.ffmpeg_process = None
    
    def push_video_to_rtmp(self) -> Optional[subprocess.Popen]:
        """
        将测试视频推送到RTMP服务器
        
        返回:
            FFmpeg进程对象或None（如果启动失败）
        """
        try:
            # 检查视频文件是否存在
            if not os.path.exists(self.test_video_path):
                logger.error(f"测试视频文件不存在: {self.test_video_path}")
                return None
            
            # FFmpeg命令
            command = [
                'ffmpeg',
                '-re',  # 实时模式
                '-i', self.test_video_path,
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-tune', 'zerolatency',
                '-c:a', 'aac',
                '-ar', '44100',
                '-f', 'flv',
                self.input_rtmp_url
            ]
            
            logger.info(f"启动FFmpeg推流: {' '.join(command)}")
            
            # 创建FFmpeg进程
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 等待一段时间确认FFmpeg启动成功
            time.sleep(2)
            
            # 检查进程是否仍在运行
            if process.poll() is not None:
                _, stderr = process.communicate()
                logger.error(f"FFmpeg推流启动失败: {stderr.decode('utf-8')}")
                return None
            
            logger.info(f"FFmpeg推流已启动，推送到: {self.input_rtmp_url}")
            return process
        except Exception as e:
            logger.error(f"启动FFmpeg推流时出错: {str(e)}")
            return None
    
    def start_processing(self) -> bool:
        """
        启动视频处理
        
        返回:
            是否成功启动处理
        """
        try:
            # API请求数据
            data = {
                "input_rtmp_url": self.input_rtmp_url,
                "output_rtmp_url": self.output_rtmp_url,
                "model_weights_path": self.model_path,
                "fps": 30,
                "width": 1280,
                "height": 720,
                "bitrate": "2000k"
            }
            
            # 发送请求
            logger.info(f"发送启动请求到API: {self.api_url}/start")
            response = requests.post(f"{self.api_url}/start", json=data)
            
            # 检查响应
            if response.status_code == 200:
                logger.info("视频处理已成功启动")
                return True
            else:
                logger.error(f"启动视频处理失败: {response.text}")
                return False
        except Exception as e:
            logger.error(f"启动视频处理时出错: {str(e)}")
            return False
    
    def monitor_metrics(self, duration: int = 60):
        """
        监控性能指标
        
        参数:
            duration: 监控持续时间（秒）
        """
        logger.info(f"开始监控性能指标，持续 {duration} 秒")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                # 获取状态
                response = requests.get(f"{self.api_url}/status")
                
                if response.status_code == 200:
                    data = response.json()
                    if data["is_processing"]:
                        metrics = data["metrics"]
                        logger.info(f"性能指标: FPS={metrics['fps']:.1f}, 延迟={metrics['latency']:.1f}ms, 检测时间={metrics['detection_time']*1000:.1f}ms")
                    else:
                        logger.warning("视频处理未运行")
                else:
                    logger.error(f"获取状态失败: {response.text}")
            except Exception as e:
                logger.error(f"监控指标时出错: {str(e)}")
            
            # 每秒更新一次
            time.sleep(1)
    
    def stop_processing(self) -> bool:
        """
        停止视频处理
        
        返回:
            是否成功停止处理
        """
        try:
            # 发送请求
            logger.info(f"发送停止请求到API: {self.api_url}/stop")
            response = requests.post(f"{self.api_url}/stop")
            
            # 检查响应
            if response.status_code == 200:
                logger.info("视频处理已成功停止")
                return True
            else:
                logger.error(f"停止视频处理失败: {response.text}")
                return False
        except Exception as e:
            logger.error(f"停止视频处理时出错: {str(e)}")
            return False
    
    def stop_ffmpeg(self):
        """停止FFmpeg进程"""
        if self.ffmpeg_process:
            try:
                logger.info("停止FFmpeg推流")
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
                logger.info("FFmpeg推流已停止")
            except Exception as e:
                logger.error(f"停止FFmpeg推流时出错: {str(e)}")
                try:
                    self.ffmpeg_process.kill()
                except:
                    pass
            
            self.ffmpeg_process = None
    
    def run_test(self, duration: int = 60):
        """
        运行完整测试
        
        参数:
            duration: 测试持续时间（秒）
        """
        try:
            # 1. 推送视频到RTMP服务器
            self.ffmpeg_process = self.push_video_to_rtmp()
            if not self.ffmpeg_process:
                logger.error("无法启动视频推流，测试失败")
                return
            
            # 2. 启动视频处理
            if not self.start_processing():
                logger.error("无法启动视频处理，测试失败")
                self.stop_ffmpeg()
                return
            
            # 3. 监控性能指标
            self.monitor_metrics(duration)
            
            # 4. 停止视频处理
            self.stop_processing()
            
            # 5. 停止视频推流
            self.stop_ffmpeg()
            
            logger.info("测试完成")
        except KeyboardInterrupt:
            logger.info("测试被用户中断")
            self.stop_processing()
            self.stop_ffmpeg()
        except Exception as e:
            logger.error(f"测试过程中出错: {str(e)}")
            self.stop_processing()
            self.stop_ffmpeg()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLO视频处理管道测试")
    parser.add_argument("--video", required=True, help="测试视频文件路径")
    parser.add_argument("--rtmp-server", default="localhost", help="RTMP服务器地址")
    parser.add_argument("--rtmp-port", type=int, default=1935, help="RTMP服务器端口")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API服务器URL")
    parser.add_argument("--model-path", default="weights/yolo11n-obb.pt", help="模型路径")
    parser.add_argument("--duration", type=int, default=60, help="测试持续时间（秒）")
    args = parser.parse_args()
    
    # 创建测试管道
    pipeline = TestPipeline(
        test_video_path=args.video,
        rtmp_server=args.rtmp_server,
        rtmp_port=args.rtmp_port,
        api_url=args.api_url,
        model_path=args.model_path
    )
    
    # 运行测试
    pipeline.run_test(args.duration)

if __name__ == "__main__":
    main() 