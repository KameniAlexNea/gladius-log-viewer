from __future__ import annotations

import gradio as gr
import argparse
from gladius_parser.app import build_ui as _build_ui

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    app = _build_ui(default_path=args.log)
    app.launch(server_name=args.host, server_port=args.port, theme=gr.themes.Base())
