import asyncio
import base64
import json
import logging
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("echo_bot")

# App
app = FastAPI(title="WebSocket Echo Bot")

# info page
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><body style="font-family:monospace;padding:2rem">
    <h2>WebSocket Echo Bot</h2>
    <p>Connect via WebSocket at <code>ws://localhost:8000/ws</code></p>
    <p>Run the client: <code>python client.py</code></p>
    </body></html>
    """

# Echo Bot logic
class EchoBot:

    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.stream_sid: str | None = None
        self.chunk_count = 0
        self.bytes_received = 0
        self.session_start = datetime.now(timezone.utc)

    # Inbound handlers
    async def handle_connected(self, msg: dict):
        log.info("Stream CONNECTED")

    async def handle_start(self, msg: dict):
        self.stream_sid = msg.get("streamSid", "unknown")
        start_meta = msg.get("start", {})
        log.info(
            "Stream START  sid=%s  encoding=%s  sampleRate=%s",
            self.stream_sid,
            start_meta.get("encoding", "?"),
            start_meta.get("sampleRate", "?"),
        )

    async def handle_media(self, msg: dict):
        media = msg.get("media", {})
        payload_b64 : str = media.get("payload", "")
        chunk_num : int = media.get("chunk", self.chunk_count)
        timestamp : str = media.get("timestamp", "")

        # Decode to  raw bytes
        raw_audio: bytes = base64.b64decode(payload_b64)
        self.bytes_received += len(raw_audio)
        self.chunk_count += 1

        log.info(
            "Received chunk #%d  bytes=%d B  total_bytes=%d B  timestamp=%s",
            chunk_num,
            len(raw_audio),
            self.bytes_received,
            timestamp,
        )

        echo_payload = base64.b64encode(raw_audio).decode("utf-8")

        await self._send_media(echo_payload)
        await self._send_mark("echo_sent")

    async def handle_stopped(self, msg: dict):
        duration = (datetime.now(timezone.utc) - self.session_start).total_seconds()
        log.info(
            "Stream STOPPED total_chunks=%d  total_bytes=%d duration=%.2f sec",
            self.chunk_count,
            self.bytes_received,
            duration,
        )

    async def _send_media(self, payload_b64: str):
        msg = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {
                "payload": payload_b64,
                "chunkNum": self.chunk_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        }
        await self.ws.send_text(json.dumps(msg))
        log.info("Sent echo chunk #%d  bytes=%d B", self.chunk_count, len(base64.b64decode(payload_b64)))

    async def _send_mark(self, mark_name: str):
        msg = {
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {
                "name": mark_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        await self.ws.send_text(json.dumps(msg))
        log.info("Sent mark: %s", mark_name)

    async def run(self):
        handlers = {
            "connected": self.handle_connected,
            "start": self.handle_start,
            "media": self.handle_media,
            "stopped": self.handle_stopped,
        }

        async for raw in self._iter_messages():
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("Received non-JSON message, skipping.")
                continue

            event = msg.get("event", "")
            if event in handlers:
                await handlers[event](msg)
            else:
                log.warning("Unknown event type: %s", event)

    async def _iter_messages(self):
        try:
            while True:
                raw = await self.ws.receive_text()
                yield raw
        except WebSocketDisconnect:
            log.info("WebSocket disconnected")
            

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log.info("Client connected : %s", websocket.client)
    bot = EchoBot(websocket)
    await bot.run()
    log.info("WebSocket connection closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)