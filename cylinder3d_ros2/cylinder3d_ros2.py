import os
import sys
import argparse
import rclpy
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType, ParameterDescriptor
from sensor_msgs.msg import PointCloud2, Imu, NavSatFix
from sensor_msgs.msg import PointField
import numpy as np
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
# from .submodules.test_semantic import points_semantic as ps
from ament_index_python import get_package_share_directory,get_package_prefix
homepath = os.path.join(os.path.dirname(os.path.dirname(get_package_prefix('cylinder3d_ros2'))), 'src', 'cylinder3d_ros2', 'cylinder3d_ros2', 'submodules')
sys.path.append(homepath)
from submodules.semantic_seg import points_semantic as ps
from submodules.dataloader_ros import SemKITTI_demo
from multiprocessing import Process, Queue
import time
import torch.multiprocessing as tmp
from torch.multiprocessing import Manager
import cfg

class Listener(Node):
    def __init__(self,name,context,dataset_config,grid_size,p):
        super().__init__(name,context=context)
        self.i = 0
        self.downsample_flag = 0
        self.scans0 = np.zeros((1,4))
        self.dtype_list = [('x', np.dtype('float32')), ('y', np.dtype('float32')), ('z', np.dtype('float32')), ('intensity', np.dtype('float32'))]
        # [('x', np.dtype('float32')), ('y', np.dtype('float32')), ('z', np.dtype('float32')), ('intensity', np.dtype('float32')), ('rgb', np.dtype('float32')), ('label', np.dtype('float32'))]
        self.subscriber = self.create_subscription(PointCloud2,
                                                  cfg.pointcloud_topic,
                                                  self.sub_callback,
                                                  10)
        self.data_preprocesser = SemKITTI_demo(grid_size=grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],)
        self.p = p
        self.get_logger().info("Listener is started, listening to %s" % cfg.pointcloud_topic)

    def sub_callback(self,msg):
        assert isinstance(msg, PointCloud2)
        data = np.frombuffer(msg.data, dtype=self.dtype_list)
        if cfg.sem_num == 2:
            if self.i == 0:
                self.scans0 = np.array(data.tolist())
                self.scans0 = np.array(self.scans0[:,0:4])
                [self.demo_grid0, self.demo_pt_fea0] = self.data_preprocesser.preprocess(self.scans0)
                self.i = 1
            else:
                scans1 = np.array(data.tolist())
                scans1 = np.array(scans1[:,0:4])
                [demo_grid1, demo_pt_fea1] = self.data_preprocesser.preprocess(scans1)
                scans = [self.scans0, scans1, self.demo_grid0, demo_grid1, self.demo_pt_fea0, demo_pt_fea1]
                if self.p.full():
                    self.p.get()
                self.p.put(scans)
                self.i = 0
        elif cfg.sem_num == 1:
            scans0 = np.array(data.tolist())
            scans0 = np.array(scans0[:,0:4])
            [demo_grid0, demo_pt_fea0] = self.data_preprocesser.preprocess(scans0)
            scans = [scans0, demo_grid0, demo_pt_fea0]
            if self.p.full():
                self.p.get()
            self.p.put(scans)

class Publisher(Node):
    def __init__(self,name,context,q):
        super().__init__(name,context=context)
        self.pub = self.create_publisher( PointCloud2, cfg.publish_topic,1)
        timer_period = 0.05
        self.color = cfg.pc_colors
        self.timer = self.create_timer(timer_period,self.timer_callback) #创建定时器
        self.q = q
        self.get_logger().info("Publisher has been started, publishing topic: %s" % (cfg.publish_topic))
    def timer_callback(self):
        if not self.q.empty():
            points = self.q.get(True)
            color_points = [self.color.get(key) for key in points[:,4].astype(int)]
            color_points = np.expand_dims(color_points,axis=1)
            points = np.append(points, color_points,axis=1)
            msg = self.msg_maker(points)
            self.pub.publish(msg)
    def msg_maker(self,points):
        header = Header()
        header.stamp = Clock().now().to_msg()
        header.frame_id = "velodyne"
        fields = [
            PointField(name='x',  offset=0, datatype=PointField.FLOAT32, count = 1),
            PointField(name='y',  offset=4, datatype=PointField.FLOAT32, count = 1),
            PointField(name='z',  offset=8, datatype=PointField.FLOAT32, count = 1),
            PointField(name='intensity',  offset=12, datatype=PointField.FLOAT32, count = 1),
            PointField(name='label',  offset=16, datatype=PointField.FLOAT32, count = 1),
            PointField(name='rgb',  offset=20, datatype=PointField.FLOAT32, count = 1)]
        msg = point_cloud2.create_cloud(header, fields, points)
        return msg

def listener_runner(dataset_config,grid_size, p):
    context1 = rclpy.context.Context() # 创建一个新的context对象
    rclpy.init(context=context1) # 使用context参数初始化rclpy
    semantic_listener = Listener('semantic_listener',context1, dataset_config, grid_size, p)
    executor = rclpy.executors.SingleThreadedExecutor(context=context1)
    try:
        rclpy.spin(semantic_listener,executor)
    except:
        pass
    semantic_listener.destroy_node()
    rclpy.shutdown(context=context1)

def publisher_runner(q):
    context2 = rclpy.context.Context() # 创建一个新的context对象
    rclpy.init(context=context2) # 使用context参数初始化rclpy
    # node2 = rclpy.create_node('semantic_publisher',context=context) # 使用context参数创建一个新的node2
    semantic_publisher = Publisher("semantic_publisher", context2, q)
    executor = rclpy.executors.SingleThreadedExecutor(context=context2)
    try:
        rclpy.spin(semantic_publisher,executor)
    except:
        pass
    semantic_publisher.destroy_node()
    rclpy.shutdown(context=context2)
    # rclpy.shutdown()

def semantic_runner(configs, p, q):
    sem_points = ps(configs)
    context = rclpy.context.Context() # 创建一个新的context对象
    rclpy.init(context=context) # 使用context参数初始化rclpy
    node = rclpy.create_node('semantic_network',context=context) # 使用context参数创建一个新的node
    # 改变程序运行的优先级
    # print(os.getpriority(os.PRIO_PROCESS, os.getpid()))
    # -20~19，-20最高，19最低
    # os.system("echo " + configs['sudo_password'] + " | sudo -S renice -5 -p " + str(os.getpid()))
    # os.setpriority(os.PRIO_PROCESS, os.getpid(), -5)


    # 向终端输出信息，使用ros2 info/debug/warn/error/fatal
    node.get_logger().info('Init ready!')
    # print("init ready!")
    while rclpy.ok(context = context):
        if not p.empty():
            start = time.time()
            scans = p.get(True)
            print("get time: ", time.time()-start)
            if cfg.sem_num == 2:
                afsem_points = sem_points.sem_points2(np.array(scans[2:4]), np.array(scans[4:]), np.array(scans[0:2]))
                q.put(afsem_points[0])
                q.put(afsem_points[1])
            elif cfg.sem_num == 1:
                afsem_points = sem_points.sem_points(scans[1], scans[2], scans[0])
                q.put(afsem_points)
            node.get_logger().info('sem time: %f' % (time.time()-start))
        # rate.sleep()
        else:
            time.sleep(0.01)

def main(args=None):
    from submodules.config.config import load_config_data
    yaml_file = cfg.yaml_file
    configs = load_config_data(homepath + yaml_file)
    dataset_config = configs['dataset_params']
    model_config = configs['model_params']
    grid_size = model_config['output_shape']
    configs['device'] = 'cuda:0'
    configs['homepath'] = homepath
    p = Manager().Queue(10)
    q = Manager().Queue()
    process = Process(target=listener_runner, args=(dataset_config, grid_size, p))
    process.start()
    process2 = Process(target=publisher_runner, args=(q, ))
    process2.start()
    time.sleep(2)
    ctx = tmp.get_context('spawn')
    process3 = ctx.Process(target=semantic_runner, args=(configs, p, q))
    process3.start()
    
    # print("init ready!")
    try:
        process.join()
        process2.join()
        process3.join()
    except KeyboardInterrupt:
        print("Keyboard Interrupt!")
        try:
            # rclpy.shutdown(context=context)
            process.terminate()
            process2.terminate()
            process3.terminate()
            process.join()
            process2.join()
            process3.join()
        except:
            pass
        return
    else:
        return


if __name__ =='__main__':
    main()
    # os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
