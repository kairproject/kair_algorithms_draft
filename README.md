<p align="center">
  <img src="https://user-images.githubusercontent.com/6107926/54689322-a0503500-4b62-11e9-87f7-aff03dbbb6dd.png" alt="KAIR"/>
</p>


<p align="center">
  <a href="https://travis-ci.org/kairproject/kair_algorithms_draft">
    <img src="https://travis-ci.org/kairproject/kair_algorithms_draft.svg?branch=master" alt="Build Status"/>
  </a>
  <a href="http://google.github.io/styleguide/pyguide.html">
    <img src="https://img.shields.io/badge/docstring-google-blue.svg" alt="Google Docstring style"/>
  </a>
  <a href=".pre-commit-config.yaml">
    <img src="https://img.shields.io/badge/pre--commit-enabled-blue.svg" alt="Pre-commit enabled"/>
  </a>
</p>


## Research Repository Structure

We follow the `.git` strategy specified [here](https://answers.ros.org/question/257855/git-strategy-for-catkin-and-package-folders/), creating packages into separate repositories.

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

## TODO

1. `launch` and `urdf` contains code for Sawyer robot. We plan to use OpenManipulator.
2. `package.xml` should be updated appropriately.
3. Get repository verified by a ROS expert.
