#!/bin/bash
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTERN="$REPO_ROOT/extern/GR00T-WholeBodyControl/gear_sonic_deploy"

cd "$EXTERN"

# GPU設定
GPU_SETTINGS="--gpus all"
TENSORRT_MOUNT="-v /home/unitree-g1/TensorRT:/opt/TensorRT:ro"

# 既存コンテナを停止
docker stop g1-deploy-dev 2>/dev/null || true
sleep 1

# コンテナが既に起動中か確認
if ! docker ps --format '{{.Names}}' | grep -q "^g1-deploy-dev$"; then
    echo "Docker コンテナを起動中..."
    docker run -d --rm \
        --name g1-deploy-dev \
        --network host \
        $GPU_SETTINGS \
        -v "$(pwd):/workspace/g1_deploy:rw" \
        $TENSORRT_MOUNT \
        -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
        -e ROS_DOMAIN_ID=0 \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        -w /workspace/g1_deploy \
        g1-deploy-dev \
        sleep infinity
    echo "コンテナ起動待機中..."
    sleep 3
fi

echo "deploy.sh を実行中..."
docker exec -it g1-deploy-dev bash -c "source scripts/setup_env.sh && bash deploy.sh --input-type zmq_manager sim"
