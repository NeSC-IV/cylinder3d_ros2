'''
Parameters of cylinder3d_ros2
'''

pointcloud_topic = "/velodyne_points"
sem_num = 1 # Simultaneous inference of the number of point cloud frames, 2 in orin paper
publish_topic = '/sem_points'
yaml_file = "/config/semantickitti32_run.yaml" # The path of the configuration file
pc_colors = {
                0: 0x000000,
                1: 0xff0000,
                10: 0x6496f5,
                11: 0x64e6f5,
                13: 0x6450fa,
                15: 0x1e3c96,
                16: 0x0000ff,
                18: 0x501eb4,
                20: 0x0000ff,
                30: 0xff1e1e,
                31: 0xff28c8,
                32: 0x961e5a,
                40: 0xff00ff,
                44: 0xff96ff,
                48: 0x4b004b,
                49: 0xaf004b,
                50: 0xffc800,
                51: 0xff7832,
                52: 0xff9600,
                60: 0x96ffaa,
                70: 0x00af00,
                71: 0x873c00,
                72: 0x96f050,
                80: 0xfff096,
                81: 0xff0000,
                99: 0x32ffff,
                252: 0x6496f5,
                253: 0xff28c8,
                254: 0xff1e1e,
                255: 0x961e5a,
                256: 0x0000ff,
                257: 0x6450fa,
                258: 0x501eb4,
                259: 0x0000ff
            }
