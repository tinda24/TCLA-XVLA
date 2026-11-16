import numpy as np
import requests


class ServerError(RuntimeError):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


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
        if response.status_code >= 400:
            raise ServerError(response.status_code, response.text)
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

    def plan(self, qpos, images):
        if len(images) != len(self.metadata["camera_names"]):
            raise ValueError("Camera count must match the checkpoint")
        payload = {
            "plan": True,
            "qpos": np.asarray(qpos, dtype=np.float32).tolist(),
        }
        payload.update(
            {
                f"image{i}": np.asarray(image, dtype=np.uint8).tolist()
                for i, image in enumerate(images)
            }
        )
        body = self._request("POST", "/act", payload)
        if "actions" in body:
            actions = np.asarray(body["actions"], dtype=np.float32)
        else:
            # A server started before chunk planning returns one action.
            actions = np.asarray(body["action"], dtype=np.float32)[None, :]
        if actions.ndim != 2 or actions.shape[1] != self.metadata["action_dim"]:
            raise ValueError("Server returned an invalid action chunk")
        return actions

    def commit(self, executed):
        try:
            self._request("POST", "/commit", {"executed": int(executed)})
        except ServerError as exc:
            if exc.status_code != 404:
                raise

    def close(self):
        self.session.close()
