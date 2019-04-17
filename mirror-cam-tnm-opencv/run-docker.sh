xhost +local:root
nvidia-docker run -it --rm --privileged -e DISPLAY \
--privileged \
-v /tmp/.X11-unix:/tmp/.X11-unix  \
-v ${PWD}:/root/project/mirror-cam-tnm-opencv    \
-w /root/project/mirror-cam-tnm-opencv \
bst/zed

