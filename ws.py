import websocket
import json
import os

runners = {}
ws_url = os.getenv('WS_URL')
if ws_url is None:
    ws_url = 'ws://127.0.0.1:3003/ws/deepseek?name=test'

def on_message(ws, message):
    if message=="pong":
        return
    try:
        data = json.loads(message)
        t = data['type']
        if t not in runners:
            ws.send(json.dumps({'id': data['id'], 'error': f"Runner {t} not found"}))
            return
        try:
            print("call", t)
            ret = runners[t](**data['data'])
            ws.send(json.dumps({'id': data['id'], 'data': ret}))
        except Exception as e:
            print(e)
            ws.send(json.dumps({'id': data['id'], 'error': str(e)}))
    except Exception as e:
        print(e)
    

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"Closed: {close_status_code} {close_msg}")

def on_open(ws):
    print("on connect")

def connect_websocket():
    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever(
        ping_interval=10,
        ping_timeout=5,
        ping_payload="ping",
        reconnect=1
    )

if __name__ == "__main__":
    print(f"ws: {ws_url}")
    # from codex import codex
    def codex(code="", max_length=128):
        return "code"
    runners['code'] = codex
    connect_websocket()