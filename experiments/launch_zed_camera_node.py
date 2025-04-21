from dataclasses import dataclass
from multiprocessing import Process

import tyro

from gello.cameras.zed_camera import ZEDCamera, gather_zed_camera_by_id, list_zed_camera_ids
from gello.zmq_core.camera_node import ZMQServerCamera


@dataclass
class Args:
    # hostname: str = "127.0.0.1"
    hostname: str = "192.168.1.110"
    camera_port: int = 5001
    zed_cam_id: int = 0000


def launch_server(port: int, camera: ZEDCamera, args: Args):
    server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting camera {camera.serial_number}  server on port {port} and host {args.hostname}")
    server.serve()


def main(args):
    zed_camera = gather_zed_camera_by_id(args.zed_cam_id)
    launch_server(args.camera_port, zed_camera, args)


if __name__ == "__main__":
    main(tyro.cli(Args))
