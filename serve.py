import argparse
from threading import Lock

from fastapi import FastAPI, HTTPException
import uvicorn

from configuration import absolute_path, load_paths


def build_app(runtime):
    app = FastAPI()
    lock = Lock()

    @app.get("/info")
    def info():
        return runtime.metadata

    @app.post("/reset")
    def reset(payload: dict):
        with lock:
            try:
                runtime.reset(
                    payload.get("task_name"),
                    payload.get("task_emb"),
                    payload.get("language_instruction"),
                )
            except (ValueError, KeyError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/act")
    def act(payload: dict):
        with lock:
            try:
                return {"action": runtime.act(payload).tolist()}
            except (ValueError, KeyError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Serve a raw or latent XVLA-TCLA policy checkpoint."
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--execute_steps", type=int)
    args = parser.parse_args()
    from inference.runtime import PolicyRuntime

    checkpoint = (
        absolute_path(args.checkpoint)
        if args.checkpoint
        else load_paths().paths.policy_checkpoint
    )
    runtime = PolicyRuntime(checkpoint, args.device, args.steps, args.execute_steps)
    uvicorn.run(build_app(runtime), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
