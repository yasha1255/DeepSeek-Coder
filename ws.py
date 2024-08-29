import websocket
import json
import time
import signal
import sys
import argparse
 
runners = {}
ws_url = 'ws://127.0.0.1:3003/ws/deepseek?name=test'

def on_message(ws, message):
    data = json.loads(message)
    if data['type'] not in runners:
        ws.send(json.dumps({'id': data['id'], 'error': f"Runner {data['type']} not found"}))
        return
    try:
        ret = runners[data['type']](**data['data'])
        print("ret", ret)
        ws.send(json.dumps({'id': data['id'], 'data': ret}))
    except Exception as e:
        print(e)
        ws.send(json.dumps({'id': data['id'], 'error': str(e)}))
    

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    if runners is None:
        return
    print(f"Closed: {close_status_code} {close_msg}")
    print("Reconnecting...")
    time.sleep(2)  # Wait for 2 seconds before attempting to reconnect
    connect_websocket()

def on_sigint(signum, frame):
    global runners
    runners = None
    print("Received SIGINT, closing gracefully...")
    sys.exit(0)


def connect_websocket():
    signal.signal(signal.SIGINT, on_sigint)
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

def get_ws_url():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--ws', type=str, required=True, help='The websocket URL')
    args = parser.parse_args()
    return args.ws_url

if __name__ == "__main__":
    ws_url = get_ws_url()
    print(f"ws: {ws_url}")
    from codex import codex
    runners['code'] = codex
    connect_websocket()