import asyncio
import base64
import json
import logging
import time
import math
import websockets

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("client")

# config
SERVER_URL   = "ws://localhost:8000/ws"
STREAM_SID   = "MXtest_stream_001"
SAMPLE_RATE  = 8000          # 8 kHz — standard Twilio telephony rate
CHUNK_MS     = 20            # 20 ms per chunk — Twilio default
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_MS / 1000)   # 160 samples
NUM_CHUNKS   = 10            # how many audio chunks to stream
TONE_HZ      = 440           # A4 tone for our fake audio  

# Audio helpers

def linear_to_ulaw(sample: int) -> int:
    ULAW_MAX = 0x1FFF
    BIAS = 0x84
    CLIP = 32635

    sample = max(-CLIP, min(CLIP, sample))
    sign = 0 if sample >= 0 else 0x80
    sample = abs(sample)
    sample += BIAS

    exp = 7
    for exp_mask in [0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100]:
        if sample & exp_mask:
            break
        exp -= 1

    mantissa = (sample >> (exp + 3)) & 0x0F
    ulaw_byte = ~(sign | (exp << 4) | mantissa) & 0xFF
    return ulaw_byte

def generate_ulaw_chunk(chunk_index: int) -> bytes:
    chunk_data = []
    for i in range(SAMPLES_PER_CHUNK):
        t = (chunk_index * SAMPLES_PER_CHUNK + i) / SAMPLE_RATE
        sample = int(32767 * math.sin(2 * math.pi * TONE_HZ * t))
        ulaw_sample = linear_to_ulaw(sample)
        chunk_data.append(ulaw_sample)
    return bytes(chunk_data)

# Client

async def run_client():
    log.info("Connecting to server at %s", SERVER_URL)

    async with websockets.connect(SERVER_URL) as ws:
        log.info("Connected to server")

        # Send CONNECTED message
        await ws.send(json.dumps({
            "event": "connected",
            "protocol": "Call",
            'version': "1.0.0",
        }))
        log.info("Sent CONNECTED")

        # Send START message
        await ws.send(json.dumps({
            "event": "start",
            "streamSid": STREAM_SID,
            "start": {
                "accountSid": "ACtest",
                "streamSid": STREAM_SID,
                "callSid": "CAtest_call",
                "tracks": ["inbound"],
                "encoding": "audio/x-mulaw",
                "sampleRate": SAMPLE_RATE,
            }
        }))
        log.info("Sent START (streamSid=%s)", STREAM_SID)

        # Streaming audio chunks
        echoes_received = 0
        marks_received  = 0

        for chunk_num in range(1, NUM_CHUNKS + 1):
            # Generate and send one audio chunk
            raw_audio  = generate_ulaw_chunk(chunk_num - 1)
            payload_b64 = base64.b64encode(raw_audio).decode()
            timestamp   = str(int(time.time() * 1000))

            await ws.send(json.dumps({
                "event": "media",
                "streamSid": STREAM_SID,
                "media": {
                    "track": "inbound",
                    "chunk": chunk_num,
                    "timestamp": timestamp,
                    "payload": payload_b64,
                }
            }))
            log.info(" Sent chunk #%d  (%d B)", chunk_num, len(raw_audio))

            # Receive echo + mark
            for _ in range(2):
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    msg = json.loads(response)
                    event = msg.get("event")

                    if event == "media":
                        echoes_received += 1
                        log.info(" Received echo chunk #%d  (%d B)", msg["media"]["chunkNum"], len(base64.b64decode(msg["media"]["payload"])))
                    elif event == "mark":
                        marks_received += 1
                        log.info(" Received mark: %s", msg["mark"]["name"])
                    else:
                        log.warning(" Received unknown message: %s", msg)

                except asyncio.TimeoutError:
                    log.error("Timeout waiting for response to chunk #%d", chunk_num)
                    break

            # Small delay between chunks (realistic pacing)
            await asyncio.sleep(CHUNK_MS / 1000)

        # Send STOP message
        await ws.send(json.dumps({
            "event": "stop",
            "streamSid": STREAM_SID,
            "stop": {
                "accountSid": "ACtest",
                "streamSid": STREAM_SID,
                "callSid": "CAtest_call"
            }
        }))
        log.info("Sent STOP")

        # Briefly wait for any final messages from server
        await asyncio.sleep(1)

    # Summary
    log.info("-" * 50)
    log.info("Session completed")
    log.info("Chunks sent: %d", NUM_CHUNKS)
    log.info("Echoes received: %d", echoes_received)
    log.info("Marks received: %d", marks_received)

    
if __name__ == "__main__":
    asyncio.run(run_client())