#!/bin/bash
set -e

echo "启动YOLO视频流处理服务..."

# 确保nginx日志和运行目录存在
mkdir -p /var/log/nginx /var/run

# 检查weights目录
if [ -z "$(ls -A /app/weights)" ]; then
    echo "警告: weights目录为空，请确保模型文件已正确挂载"
else
    echo "检测到weights目录中的文件:"
    ls -la /app/weights
fi

# 复制nginx配置到默认位置
echo "复制nginx配置到默认位置..."
mkdir -p /etc/nginx
cp /app/nginx.conf /etc/nginx/nginx.conf

# 启动nginx
echo "启动Nginx服务..."
/usr/sbin/nginx

# 等待nginx启动
sleep 2
if ! pgrep -x "nginx" > /dev/null; then
    echo "Nginx启动失败，查看错误日志:"
    cat /var/log/nginx/error.log
    exit 1
fi

echo "Nginx已成功启动"

# 启动API服务器
echo "启动API服务器..."
cd /app
python3 api_server.py &

# 等待API服务器启动
sleep 3

# 保持容器运行
echo "所有服务已启动，按Ctrl+C退出"
tail -f /var/log/nginx/error.log 