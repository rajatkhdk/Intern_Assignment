# Asynchronous WebSocket Echo Bot

A FastAPI-based asyunchronous WebSocket server that simulate real-time telephony-style audio streaming.

# Features:

- Bidirectional Websocket communication
- Asyunc audio streaming simulation
- Real-time audio responses
- Response acknowledgement system
- Lightweight FastAPI server

# System workflow

1. Client sends Audio Chunks:
the client simulates streaming sudio data in chunks.

2. Server Receives Stream:
The FastAPI WebSocket server receives audio frames asynchronously.

3. Echo Response:
The server sends responses back to the client.

4.  Acknowledgement:
The client receives confirmation messages such as:
"echo_sent"

# Running the Project

### Change directory
```bash
cd websocket_echobot
```

### Start server
```bash
python server.py
```

### Run client
```bash
python client.py
```