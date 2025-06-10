FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装Nginx和RTMP模块
RUN apt-get update && apt-get install -y \
    nginx \
    libnginx-mod-rtmp \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 创建nginx日志和运行目录
RUN mkdir -p /var/log/nginx /var/run /etc/nginx

# 创建模型目录
RUN mkdir -p /app/weights

# 复制项目文件，但排除weights目录
COPY requirements.txt /app/
COPY *.py /app/
COPY nginx.conf /etc/nginx/nginx.conf
COPY README.md /app/
COPY todo.md /app/
COPY docker_entrypoint.sh /app/

# 设置入口点脚本为可执行
RUN chmod +x /app/docker_entrypoint.sh

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 1935 8000 8080

# 创建一个空的weights目录，用于挂载
VOLUME ["/app/weights"]

# 设置启动命令
ENTRYPOINT ["/app/docker_entrypoint.sh"] 