# -*- coding: utf-8 -*-
"""Collect demo data with jacobian based control method.

- Author: DH Kim
- Contact: kdh0429@snu.ac.kr
"""

import rospy
from config.demo.open_manipulator.reacher_v0 import config as cfg
from demo.open_manipulator.open_manipulator_demo_collector import DemoCollector


def main():
    """Main."""
    # env initialization

    try:
        collector = DemoCollector(cfg)
        collector.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
