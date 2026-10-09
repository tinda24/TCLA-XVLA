from inference.client import PolicyClient


class FrankaPolicyClient(PolicyClient):
    def __init__(self, server_url="http://127.0.0.1:8000", timeout=60):
        super().__init__(server_url, timeout)
        if self.metadata["dataset"] != "franka" or self.metadata["action_dim"] != 8:
            raise ValueError("The server must load a Franka checkpoint")

    def step(self, qpos, front_rgb, wrist_rgb):
        images = {"front_cam": front_rgb, "wrist_cam": wrist_rgb}
        return self.act(qpos, [images[name] for name in self.metadata["camera_names"]])
