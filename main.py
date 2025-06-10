import os
import argparse
import logging
import subprocess
import time
import signal
import sys
from typing import List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("YOLO-Main")

# 全局进程列表
processes = []

def start_nginx(config_path: str) -> Optional[subprocess.Popen]:
    """
    启动Nginx RTMP服务器
    
    参数:
        config_path: Nginx配置文件路径
        
    返回:
        Nginx进程对象或None（如果启动失败）
    """
    try:
        logger.info(f"使用配置文件启动Nginx: {config_path}")
        
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            logger.error(f"Nginx配置文件不存在: {config_path}")
            return None
        
        # 获取配置文件的绝对路径
        abs_config_path = os.path.abspath(config_path)
        logger.info(f"Nginx配置文件绝对路径: {abs_config_path}")
        
        # 启动Nginx，使用-c参数指定配置文件的绝对路径
        process = subprocess.Popen(
            ["nginx", "-c", abs_config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待一段时间确认Nginx启动成功
        time.sleep(2)
        
        # 检查进程是否仍在运行
        if process.poll() is not None:
            _, stderr = process.communicate()
            stderr_text = stderr.decode('utf-8') if stderr else "未知错误"
            logger.error(f"Nginx启动失败: {stderr_text}")
            return None
        
        logger.info("Nginx RTMP服务器已启动")
        return process
    except Exception as e:
        logger.error(f"启动Nginx时出错: {str(e)}")
        return None

def start_api_server() -> Optional[subprocess.Popen]:
    """
    启动API服务器
    
    返回:
        API服务器进程对象或None（如果启动失败）
    """
    try:
        logger.info("启动API服务器")
        
        # 启动API服务器
        process = subprocess.Popen(
            ["python", "api_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待一段时间确认API服务器启动成功
        time.sleep(3)
        
        # 检查进程是否仍在运行
        if process.poll() is not None:
            _, stderr = process.communicate()
            stderr_text = stderr.decode('utf-8') if stderr else "未知错误"
            logger.error(f"API服务器启动失败: {stderr_text}")
            return None
        
        logger.info("API服务器已启动")
        return process
    except Exception as e:
        logger.error(f"启动API服务器时出错: {str(e)}")
        return None

def stop_processes(process_list: List[subprocess.Popen]):
    """
    停止所有进程
    
    参数:
        process_list: 进程列表
    """
    logger.info("停止所有进程")
    
    for process in process_list:
        try:
            if process.poll() is None:  # 如果进程仍在运行
                process.terminate()
                process.wait(timeout=5)
        except Exception as e:
            logger.error(f"停止进程时出错: {str(e)}")
            try:
                process.kill()  # 强制终止
            except:
                pass
    
    # 确保Nginx完全停止
    try:
        subprocess.run(["nginx", "-s", "stop"], check=False)
    except:
        pass
    
    logger.info("所有进程已停止")

def signal_handler(sig, frame):
    """信号处理函数，用于捕获Ctrl+C"""
    logger.info("接收到中断信号，正在停止...")
    stop_processes(processes)
    sys.exit(0)

def check_weights_directory():
    """检查weights目录是否存在，不存在则创建"""
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        logger.info(f"创建权重目录: {weights_dir}")
        os.makedirs(weights_dir)
    
    # 检查是否有权重文件
    weights_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
    if not weights_files:
        logger.warning("weights目录中没有发现模型权重文件 (.pt)，请确保在使用前下载并放置模型文件")
    else:
        logger.info(f"发现以下模型权重文件: {', '.join(weights_files)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLO视频流处理应用")
    parser.add_argument("--nginx-config", default="nginx.conf", help="Nginx配置文件路径")
    parser.add_argument("--no-nginx", action="store_true", help="不启动Nginx服务器")
    parser.add_argument("--skip-nginx", action="store_true", help="跳过Nginx检查，适用于Windows系统")
    args = parser.parse_args()
    
    global processes
    
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 检查weights目录
    check_weights_directory()
    
    try:
        # 启动Nginx RTMP服务器（如果需要）
        if not args.no_nginx and not args.skip_nginx:
            nginx_process = start_nginx(args.nginx_config)
            if nginx_process:
                processes.append(nginx_process)
            else:
                logger.error("无法启动Nginx，应用将退出")
                return 1
        elif args.skip_nginx:
            logger.info("跳过Nginx启动（--skip-nginx参数）")
        
        # 启动API服务器
        api_process = start_api_server()
        if api_process:
            processes.append(api_process)
        else:
            logger.error("无法启动API服务器，应用将退出")
            stop_processes(processes)
            return 1
        
        logger.info("所有服务已启动，按Ctrl+C退出")
        
        # 保持主程序运行
        while True:
            time.sleep(1)
            
            # 检查进程是否仍在运行
            for i, process in enumerate(processes[:]):
                if process.poll() is not None:
                    logger.error(f"进程 {i} 意外退出，返回码: {process.returncode}")
                    processes.remove(process)
            
            # 如果所有进程都退出了，退出主程序
            if not processes:
                logger.error("所有进程已退出，应用将退出")
                return 1
    except KeyboardInterrupt:
        logger.info("接收到键盘中断，正在停止...")
    except Exception as e:
        logger.error(f"运行时出错: {str(e)}")
    finally:
        stop_processes(processes)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 