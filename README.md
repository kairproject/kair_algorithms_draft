# Research Repository Structure

We follow the `.git` strategy specified [here](https://answers.ros.org/question/257855/git-strategy-for-catkin-and-package-folders/), creating packages into separate repositories.

These are repositories checked to set this structure:
- [cadrl_ros](https://github.com/mfe7/cadrl_ros)
- [ros_best_practices](https://github.com/leggedrobotics/ros_best_practices)

## TODO

1. Add [medipixel/reinforcement_learning_examples](https://github.com/medipixel/reinforcement_learning_examples) to `/scripts/`.
2. `launch` and `urdf` contains code for Sawyer robot. We plan to use OpenManipulator.
3. `package.xml` should be updated appropriately.
4. Get repository verified by a ROS expert.

## Repository Structure

```
+ launch/         -
+ msg/            - These are message descriptions for ROS.
+ scripts/        - This directory contains RL algorithms.
+ urdf/           - This package contains a C++ parser for the Unified Robot Description Format (URDF), 
- .flake8         - This file specifies what rules should be enforced via flake8.
- .gitignore      - This file specifies which folders and files to ignore in Git.
- CMakeLists.txt  - This file specifies behavior of CMake.
- package.xml     - This file specifies the ROS package.
- README.md       - You are here
```

- **roslaunch/** [http://wiki.ros.org/roslaunch/XML](http://wiki.ros.org/roslaunch/XML).
- **msg/** [http://wiki.ros.org/msg](http://wiki.ros.org/msg)
- **urdf/** [http://wiki.ros.org/urdf](http://wiki.ros.org/urdf)
- **CMakeLists.txt** [http://wiki.ros.org/catkin/CMakeLists.txt](http://wiki.ros.org/catkin/CMakeLists.txt)
- **package.xml** [http://wiki.ros.org/catkin/package.xml](http://wiki.ros.org/catkin/package.xml)
