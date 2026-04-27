"""
实验6 基于Python的本地大模型部署 - Web对话界面
模型: MING-1.8B (明医中文医疗问诊大模型)
架构: Qwen2ForCausalLM
功能: 流式回复 / 多轮对话 / 历史记录管理
"""

import torch
import json
import os
import time
import uuid
from threading import Thread
from queue import Queue
from flask import Flask, render_template, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

app = Flask(__name__)

MODEL_PATH = r"d:\桌面\实验6\MING-1.8B"
HISTORY_FILE = r"d:\桌面\实验6\chat_history.json"


class MedicalChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.conversations = []
        self.device = "cpu"
        self.load_history()

    def load_history(self):
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    if data and isinstance(data[0], dict) and "role" in data[0]:
                        if data:
                            conv = {
                                "id": self._gen_id(),
                                "title": "历史对话",
                                "created_at": time.time(),
                                "messages": data
                            }
                            self.conversations = [conv]
                    else:
                        self.conversations = data
                else:
                    self.conversations = []
            else:
                self.conversations = []
        except Exception as e:
            print(f"加载历史记录失败: {e}")
            self.conversations = []

    def save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存历史记录失败: {e}")

    def _gen_id(self):
        return uuid.uuid4().hex[:12]

    def _timestamp(self):
        return time.strftime("%H:%M")

    def create_conversation(self, title="新对话"):
        conv = {
            "id": self._gen_id(),
            "title": title,
            "created_at": time.time(),
            "messages": []
        }
        self.conversations.insert(0, conv)
        self.save_history()
        return conv

    def add_message(self, conv_id, role, content):
        for conv in self.conversations:
            if conv["id"] == conv_id:
                conv["messages"].append({
                    "role": role,
                    "content": content,
                    "time": self._timestamp()
                })
                self.save_history()
                return
        conv = self.create_conversation(content[:30] if role == "user" else "新对话")
        conv["id"] = conv_id
        conv["messages"].append({
            "role": role,
            "content": content,
            "time": self._timestamp()
        })
        self.save_history()

    def get_conversations(self):
        return sorted(self.conversations, key=lambda c: c.get("created_at", 0), reverse=True)

    def get_conversation(self, conv_id):
        for c in self.conversations:
            if c["id"] == conv_id:
                return c
        return None

    def delete_conversation(self, conv_id):
        self.conversations = [c for c in self.conversations if c["id"] != conv_id]
        self.save_history()

    def load_model(self):
        if self.is_loaded:
            return True
        try:
            print("正在加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用设备: {device}")
            print("正在加载模型...")

            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

            if hasattr(self.model, "device"):
                self.device = str(self.model.device)
            elif hasattr(self.model, "model") and hasattr(self.model.model, "device"):
                self.device = str(self.model.model.device)
            else:
                self.device = device

            print(f"模型实际设备: {self.device}")
            self.is_loaded = True
            print("模型加载完成!")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def generate_stream(self, user_input, conv_id, max_new_tokens=512):
        if not self.is_loaded:
            yield json.dumps({"error": "模型未加载，请稍候..."})
            return

        self.add_message(conv_id, "user", user_input)

        conv = self.get_conversation(conv_id)
        messages = [{"role": "system", "content": "你是一个专业的医疗助手，请用中文回答用户的医疗问题。回答要专业、准确、简洁。"}]
        if conv:
            messages.extend(conv["messages"][-10:])

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        response_text = ""
        for chunk in streamer:
            if chunk:
                response_text += chunk
                yield json.dumps({"token": chunk})

        self.add_message(conv_id, "assistant", response_text)
        yield json.dumps({"done": True})


chatbot = MedicalChatbot()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded": chatbot.is_loaded,
        "cuda_available": torch.cuda.is_available(),
        "device": chatbot.device if chatbot.is_loaded else ("cuda" if torch.cuda.is_available() else "cpu"),
    })


@app.route("/api/conversations")
def api_conversations():
    return jsonify({"conversations": chatbot.get_conversations()})


@app.route("/api/conversations/<conv_id>")
def api_conversation(conv_id):
    conv = chatbot.get_conversation(conv_id)
    if conv:
        return jsonify({"conversation": conv})
    return jsonify({"error": "对话不存在"}), 404


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    user_input = data.get("message", "").strip()
    conv_id = data.get("conversation_id", "")
    if not user_input:
        return jsonify({"error": "输入不能为空"}), 400
    if not conv_id:
        conv = chatbot.create_conversation(user_input[:30])
        conv_id = conv["id"]

    def generate():
        for chunk in chatbot.generate_stream(user_input, conv_id):
            yield f"data: {chunk}\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def api_delete_conversation(conv_id):
    chatbot.delete_conversation(conv_id)
    return jsonify({"status": "success"})


@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    chatbot.conversations = []
    chatbot.save_history()
    return jsonify({"status": "success", "message": "对话历史已清除"})


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    success = chatbot.load_model()
    if success:
        return jsonify({"status": "success", "message": "模型加载成功", "device": chatbot.device})
    else:
        return jsonify({"status": "error", "message": "模型加载失败"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("MING-1.8B 医疗大模型 Web 对话系统")
    print("=" * 60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    print("\n正在自动加载模型，请稍候...")
    chatbot.load_model()

    print("\n正在启动Web服务器...")
    print("请在浏览器中访问: http://localhost:5000")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False)
