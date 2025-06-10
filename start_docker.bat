@echo off
echo ===== YOLO视频流处理项目Docker部署脚本 =====

REM 检查weights目录是否存在
if not exist weights mkdir weights
echo 已创建weights目录，请确保将yolo11n-obb.pt文件放入该目录

REM 检查weights/yolo11n-obb.pt文件是否存在
if not exist weights\yolo11n-obb.pt (
    echo 警告: weights\yolo11n-obb.pt文件不存在
    echo 请先下载YOLO11n-OBB模型权重文件并放入weights目录
    echo 下载完成后再次运行此脚本
    pause
    exit /b 1
)

REM 检查Docker是否安装
docker --version > nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: Docker未安装或未运行
    echo 请安装Docker Desktop并确保它已启动
    pause
    exit /b 1
)

REM 启动Docker Compose
echo 正在启动Docker容器...
docker-compose up -d

if %errorlevel% neq 0 (
    echo 错误: Docker启动失败
    pause
    exit /b 1
)

echo Docker容器已成功启动
echo 可以通过以下命令查看日志:
echo docker-compose logs -f
echo.
echo 可以通过以下命令停止服务:
echo docker-compose down

pause 