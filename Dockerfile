FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
ENV ROS_DISTRO kinetic
ENV HOME=/root
ENV CATKIN_WS=/root/catkin_ws

SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get -y upgrade && apt-get -y install git wget vim

# install ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y ros-${ROS_DISTRO}-desktop-full ros-${ROS_DISTRO}-rqt-*
RUN rosdep init && rosdep update
RUN apt install -y python-rosinstall python-rosinstall-generator python-wstool build-essential
RUN apt-get install -y ros-${ROS_DISTRO}-ros-controllers ros-${ROS_DISTRO}-gazebo* ros-${ROS_DISTRO}-moveit* ros-${ROS_DISTRO}-industrial-core
RUN rm -rf /var/lib/apt/lists/*

# ROS setting
RUN mkdir -p $CATKIN_WS/src
WORKDIR $CATKIN_WS
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> $HOME/.bashrc
RUN echo "export PYTHONPATH=/opt/ros/${ROS_DISTRO}/lib/python2.7/dist-packages" >> $HOME/.bashrc
RUN echo "source $CATKIN_WS/devel/setup.bash" >> $HOME/.bashrc
RUN echo "export ROS_MASTER_URI=http://localhost:11311" >> $HOME/.bashrc
RUN echo "export ROS_HOSTNAME=localhost" >> $HOME/.bashrc
RUN source $HOME/.bashrc

# clone ROS packages
RUN cd src/ && \
	git clone -b ${ROS_DISTRO}-devel https://github.com/kairproject/DynamixelSDK.git && \
	git clone -b ${ROS_DISTRO}-devel https://github.com/kairproject/dynamixel-workbench.git && \
	git clone -b ${ROS_DISTRO}-devel https://github.com/kairproject/dynamixel-workbench-msgs.git && \
	git clone -b ${ROS_DISTRO}-devel https://github.com/kairproject/open_manipulator.git && \
	git clone -b ${ROS_DISTRO}-devel https://github.com/kairproject/open_manipulator_msgs.git && \
	git clone -b ${ROS_DISTRO}-devel https://github.com/kairproject/open_manipulator_simulations.git && \
	git clone -b ${ROS_DISTRO}-devel https://github.com/kairproject/robotis_manipulator.git && \
	git clone https://github.com/kairproject/kair_algorithms_draft.git

RUN cd src/DynamixelSDK/python && python setup.py install

# install pip
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python2.7 get-pip.py

# gym requirements
RUN apt-get update && apt-get install -y python3-opengl zlib1g-dev libjpeg-dev patchelf \
	cmake swig libboost-all-dev libsdl2-dev libosmesa6-dev xvfb ffmpeg

# install repository requirements
RUN apt-get remove -y python-psutil
RUN cd src/kair_algorithms_draft/scripts && python2.7 -m pip install -r requirements.txt
RUN python2.7 -m pip install gym['Box2d']

# permission setting
RUN chmod +x -R src/kair_algorithms_draft

# catkin_make
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make

ENTRYPOINT ["/root/catkin_ws/src/kair_algorithms_draft/docker_train.sh"]
