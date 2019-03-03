FROM ubuntu:16.04
ENV HOME /root
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get -y upgrade && apt-get -y install git wget vim

# install ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y ros-kinetic-desktop-full ros-kinetic-rqt-*
RUN rosdep init
RUN rosdep update
RUN apt install -y python-rosinstall python-rosinstall-generator python-wstool build-essential
RUN apt-get install -y ros-kinetic-ros-controllers ros-kinetic-gazebo* ros-kinetic-moveit* ros-kinetic-industrial-core

# ROS setting
RUN mkdir -p $HOME/catkin_ws/src
WORKDIR $HOME/catkin_ws/
RUN echo "source /opt/ros/kinetic/setup.bash" >> $HOME/.bashrc
RUN echo "export PYTHONPATH=/opt/ros/kinetic/lib/python2.7/dist-packages" >> $HOME/.bashrc
RUN echo "source $HOME/catkin_ws/devel/setup.bash" >> $HOME/.bashrc
RUN echo "export ROS_MASTER_URI=http://localhost:11311" >> $HOME/.bashrc
RUN echo "export ROS_HOSTNAME=localhost" >> $HOME/.bashrc
RUN source $HOME/.bashrc

# clone ROS packages
RUN cd src/ && \
	git clone https://github.com/kairproject/DynamixelSDK.git && \
	git clone https://github.com/kairproject/dynamixel-workbench.git && \
	git clone https://github.com/kairproject/dynamixel-workbench-msgs.git && \
	git clone https://github.com/kairproject/open_manipulator.git && \
	git clone https://github.com/kairproject/open_manipulator_msgs.git && \
	git clone https://github.com/kairproject/open_manipulator_simulations.git && \
	git clone https://github.com/kairproject/robotis_manipulator.git

RUN cd $HOME/catkin_ws/src/DynamixelSDK/python && python setup.py install

# permission setting
COPY ./modud_kair_control $HOME/catkin_ws/src/modud_kair_control
RUN chmod -R +x $HOME/catkin_ws/src/modud_kair_control/
