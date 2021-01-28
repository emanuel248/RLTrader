#!/usr/bin/env bash

set -e

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MEM=$(cat /proc/meminfo | grep 'MemTotal:' |  awk '{ print $2 }')
CPUS=$(cat /proc/cpuinfo | grep -P 'processor.+[0-7]+' | wc -l)

MEM_LIMIT=$((MEM/4*3))
CPU_LIMIT=$((CPUS/4*2))

if [ $CPU_LIMIT == 0 ];then
    CPU_LIMIT=1
fi

N="trader-rl-gpu"
docker build --tag $N -f "$CWD/Dockerfile" "$CWD"

echo "CWD: $CWD - Procs: $CPU_LIMIT Memory: ${MEM_LIMIT}bytes"
docker run \
    --interactive \
    --tty \
    --rm \
    --cpus "${CPU_LIMIT}" \
    --runtime=nvidia \
    --volume "${CWD}":/code \
    --user=$(id -u) \
    -e QT_X11_NO_MITSHM=1 \
    -e DISPLAY=$DISPLAY \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --network host -it \
    "$N" \
    bash $@
