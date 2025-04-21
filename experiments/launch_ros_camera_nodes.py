from dataclasses import dataclass
from multiprocessing import Process

import tyro

from gello.cameras.ros_camera import ROSCamera
from gello.zmq_core.camera_node import ZMQServerCamera


@dataclass
class Args:
    # hostname: str = "127.0.0.1"
    hostname: str = "192.168.1.2"
    port: int = 5000
    depth_image_topic_name: str = "/depth/image_raw"
    rgb_image_topic_name: str = "/rgb/image_raw"
   

def launch_server(port: int, args: Args):
    camera = ROSCamera(depth_image_topic_name=args.depth_image_topic_name,
                       rgb_image_topic_name=args.rgb_image_topic_name)
    server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting camera server on port {port}")
    server.serve()


def main(args):

    camera_port = args.port
    camera_servers = []
    print(f"Launching cameras {args.rgb_image_topic_name} and {args.depth_image_topic_name} on port {camera_port}")
    camera_servers.append(
            Process(target=launch_server, args=(camera_port, args))
        )

    for server in camera_servers:
        server.start()


if __name__ == "__main__":
    print(" calling main")
    main(tyro.cli(Args))
