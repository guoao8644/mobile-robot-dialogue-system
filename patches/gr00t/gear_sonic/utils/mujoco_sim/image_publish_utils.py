"""Offloads rendered MuJoCo camera images to a subprocess via shared memory + ZMQ."""

import multiprocessing as mp
from multiprocessing import shared_memory
import time
from typing import Any, Dict

import numpy as np

from gear_sonic.utils.mujoco_sim.sensor_server import ImageMessageSchema, SensorServer


def get_multiprocessing_info(verbose: bool = True):
    """Get information about multiprocessing start methods"""

    if verbose:
        print(f"Available start methods: {mp.get_all_start_methods()}")
    return mp.get_start_method()


class ImagePublishProcess:
    """Subprocess for publishing images using shared memory and ZMQ"""

    def __init__(
        self,
        camera_configs: Dict[str, Any],
        image_dt: float,
        zmq_port: int = 5555,
        start_method: str = "spawn",
        verbose: bool = False,
    ):
        self.camera_configs = camera_configs
        self.image_dt = image_dt
        self.zmq_port = zmq_port
        self.verbose = verbose
        self.shared_memory_blocks = {}
        self.shared_memory_info = {}
        self.process = None

        self.mp_context = mp.get_context(start_method)
        if self.verbose:
            print(f"Using multiprocessing context: {start_method}")

        self.stop_event = self.mp_context.Event()
        self.data_ready_event = self.mp_context.Event()

        self.stop_event.clear()
        self.data_ready_event.clear()
        self._active_cam_shm = shared_memory.SharedMemory(create=True, size=32)
        self._active_cam_arr = np.ndarray((32,), dtype=np.uint8, buffer=self._active_cam_shm.buf)
        self._active_cam_arr[:] = 0
        self._robot_vis_shm = shared_memory.SharedMemory(create=True, size=1)
        self._robot_vis_arr = np.ndarray((1,), dtype=np.uint8, buffer=self._robot_vis_shm.buf)
        self._robot_vis_arr[:] = 0

        for camera_name, camera_config in camera_configs.items():
            height = camera_config["height"]
            width = camera_config["width"]
            size = height * width * 3

            shm = shared_memory.SharedMemory(create=True, size=size)
            self.shared_memory_blocks[camera_name] = shm
            self.shared_memory_info[camera_name] = {
                "name": shm.name,
                "size": size,
                "shape": (height, width, 3),
                "dtype": np.uint8,
            }

    def start_process(self):
        """Start the image publishing subprocess"""
        self.process = self.mp_context.Process(
            target=self._image_publish_worker,
            args=(
                self.shared_memory_info,
                self.image_dt,
                self.zmq_port,
                self.stop_event,
                self.data_ready_event,
                self.verbose,
                self._active_cam_shm.name,
                self._robot_vis_shm.name,
            ),
        )
        self.process.start()

    def update_shared_memory(self, render_caches: Dict[str, np.ndarray]):
        """Update shared memory with new rendered images"""
        # active camera 名を書き込む
        active_cams = [k.replace("_image", "") for k in render_caches.keys() if k.endswith("_image")]
        if active_cams:
            name_bytes = active_cams[0].encode()[:31]
            self._active_cam_arr[:] = 0
            self._active_cam_arr[:len(name_bytes)] = list(name_bytes)
        # robot_visible フラグを書き込む
        self._robot_vis_arr[0] = 1 if render_caches.get("robot_in_user_view", False) else 0
        images_updated = 0
        for camera_name in self.camera_configs.keys():
            image_key = f"{camera_name}_image"
            if image_key in render_caches:
                image = render_caches[image_key]

                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)

                shm = self.shared_memory_blocks[camera_name]
                shared_array = np.ndarray(
                    self.shared_memory_info[camera_name]["shape"],
                    dtype=self.shared_memory_info[camera_name]["dtype"],
                    buffer=shm.buf,
                )

                np.copyto(shared_array, image)
                images_updated += 1

        if images_updated > 0:
            self.data_ready_event.set()

    def stop(self):
        """Stop the image publishing subprocess"""
        self.stop_event.set()

        if self.process and self.process.is_alive():
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join()

        for shm_obj, name in [(self._active_cam_shm, 'active_cam'), (self._robot_vis_shm, 'robot_vis')]:
            try:
                shm_obj.close()
                shm_obj.unlink()
            except Exception:
                pass
        for camera_name, shm in self.shared_memory_blocks.items():
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                print(f"Warning: Failed to cleanup shared memory for {camera_name}: {e}")

        self.shared_memory_blocks.clear()

    @staticmethod
    def _image_publish_worker(
        shared_memory_info, image_dt, zmq_port, stop_event, data_ready_event, verbose,
        active_cam_shm_name=None, robot_vis_shm_name=None
    ):
        """Worker function that runs in the subprocess"""
        try:
            sensor_server = SensorServer()
            sensor_server.start_server(port=zmq_port)

            shared_arrays = {}
            shm_blocks = {}
            for camera_name, info in shared_memory_info.items():
                shm = shared_memory.SharedMemory(name=info["name"])
                shm_blocks[camera_name] = shm
                shared_arrays[camera_name] = np.ndarray(
                    info["shape"], dtype=info["dtype"], buffer=shm.buf
                )

            # active_cam と robot_vis の shared memory を開く
            ac_shm = ac_arr = rv_shm = rv_arr = None
            if active_cam_shm_name:
                try:
                    ac_shm = shared_memory.SharedMemory(name=active_cam_shm_name)
                    ac_arr = np.ndarray((32,), dtype=np.uint8, buffer=ac_shm.buf)
                except Exception: pass
            if robot_vis_shm_name:
                try:
                    rv_shm = shared_memory.SharedMemory(name=robot_vis_shm_name)
                    rv_arr = np.ndarray((1,), dtype=np.uint8, buffer=rv_shm.buf)
                except Exception: pass

            print(
                f"Image publishing subprocess started with {len(shared_arrays)} cameras "
                f"on ZMQ port {zmq_port}"
            )

            loop_count = 0
            last_data_time = time.time()

            while not stop_event.is_set():
                loop_count += 1

                timeout = min(image_dt, 0.05)
                data_available = data_ready_event.wait(timeout=timeout)

                current_time = time.time()

                if data_available:
                    data_ready_event.clear()
                    if loop_count % 50 == 0:
                        print("Image publish frequency: ", 1 / (current_time - last_data_time))
                    last_data_time = current_time

                    try:
                        from gear_sonic.utils.mujoco_sim.sensor_server import ImageUtils

                        # active camera だけ送信
                        if ac_arr is not None:
                            active_name = bytes(ac_arr).rstrip(b'\x00').decode('utf-8', errors='ignore')
                        else:
                            active_name = None
                        if active_name and active_name in shared_arrays:
                            image_copies = {active_name: shared_arrays[active_name].copy()}
                        else:
                            image_copies = {name: arr.copy() for name, arr in shared_arrays.items()}

                        message_dict = {
                            "images": image_copies,
                            "timestamps": {name: current_time for name in image_copies.keys()},
                        }

                        image_msg = ImageMessageSchema(
                            timestamps=message_dict.get("timestamps"),
                            images=message_dict.get("images", None),
                        )

                        serialized_data = image_msg.serialize()

                        for camera_name, image_copy in image_copies.items():
                            serialized_data[f"{camera_name}"] = ImageUtils.encode_image(image_copy)

                        # robot_visible フラグを付加
                        if rv_arr is not None:
                            serialized_data["robot_in_user_view"] = bool(rv_arr[0])
                        sensor_server.send_message(serialized_data)

                    except Exception as e:
                        print(f"Error publishing images: {e}")

                if not data_available:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            print("Image publisher interrupted by user")
        finally:
            try:
                for shm in shm_blocks.values():
                    shm.close()
                for s in [ac_shm, rv_shm]:
                    if s:
                        try: s.close()
                        except: pass
                sensor_server.stop_server()
            except Exception as e:
                print(f"Error during subprocess cleanup: {e}")
