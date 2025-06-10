from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import os
from typing import Dict, Optional
from video_processor import VideoProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("API-Server")

# 创建FastAPI应用
app = FastAPI(title="YOLO视频流处理API")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局视频处理器实例
video_processor = None

# 请求模型
class ProcessorConfig(BaseModel):
    input_rtmp_url: str
    output_rtmp_url: str
    model_weights_path: str
    fps: Optional[int] = 30
    width: Optional[int] = 1280
    height: Optional[int] = 720
    bitrate: Optional[str] = "2000k"

@app.get("/")
async def root():
    """API根路径，返回基本信息"""
    return {
        "status": "running",
        "service": "YOLO视频流处理API",
        "version": "1.0.0"
    }

@app.post("/start")
async def start_processing(config: ProcessorConfig):
    """启动视频处理"""
    global video_processor
    
    try:
        # 检查模型文件是否存在
        if not os.path.exists(config.model_weights_path):
            raise HTTPException(status_code=400, detail=f"模型文件不存在: {config.model_weights_path}")
        
        # 如果已有处理器在运行，先停止它
        if video_processor is not None:
            video_processor.stop()
        
        # 创建新的处理器
        video_processor = VideoProcessor(
            input_rtmp_url=config.input_rtmp_url,
            output_rtmp_url=config.output_rtmp_url,
            model_weights_path=config.model_weights_path,
            fps=config.fps,
            width=config.width,
            height=config.height,
            bitrate=config.bitrate
        )
        
        # 启动处理
        video_processor.start()
        
        return {"status": "success", "message": "视频处理已启动"}
    except Exception as e:
        logger.error(f"启动视频处理时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动视频处理失败: {str(e)}")

@app.post("/stop")
async def stop_processing():
    """停止视频处理"""
    global video_processor
    
    if video_processor is None:
        return {"status": "warning", "message": "没有正在运行的视频处理器"}
    
    try:
        video_processor.stop()
        video_processor = None
        return {"status": "success", "message": "视频处理已停止"}
    except Exception as e:
        logger.error(f"停止视频处理时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"停止视频处理失败: {str(e)}")

@app.get("/status")
async def get_status():
    """获取处理状态和性能指标"""
    global video_processor
    
    if video_processor is None:
        return {
            "status": "stopped",
            "is_processing": False,
            "metrics": None
        }
    
    try:
        metrics = video_processor.get_metrics()
        return {
            "status": "running",
            "is_processing": True,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"获取状态时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

# 主入口
if __name__ == "__main__":
    # 启动Uvicorn服务器
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    ) 