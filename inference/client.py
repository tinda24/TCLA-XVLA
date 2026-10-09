import numpy as np
import requests


class PolicyClient:
    def __init__(self, server_url="http://127.0.0.1:8000", timeout=60):
        self.url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.metadata = self._request("GET", "/info")

    def _request(self, method, endpoint, payload=None):
        response = self.session.request(
            method, self.url + endpoint, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def reset(self, task_name, task_emb=None, language_instruction=None):
        payload = {"task_name": task_name, "language_instruction": language_instruction}
        if task_emb is not None:
            payload["task_emb"] = np.asarray(task_emb, dtype=np.float32).tolist()
        self._request("POST", "/reset", payload)

    def act(self, qpos, images):
        if len(images) != len(self.metadata["camera_names"]):
            raise ValueError("Camera count must match the checkpoint")
        payload = {"qpos": np.asarray(qpos, dtype=np.float32).tolist()}
        payload.update(
            {
                f"image{i}": np.asarray(image, dtype=np.uint8).tolist()
                for i, image in enumerate(images)
            }
        )
        return np.asarray(
            self._request("POST", "/act", payload)["action"], dtype=np.float32
        )

    def close(self):
        self.session.close()
