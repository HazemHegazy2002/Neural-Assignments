import os
import sys
import threading
import webbrowser
import http.server
import socketserver

PORT = 5050

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=get_base_dir(), **kwargs)
    def log_message(self, format, *args):
        pass

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()

def main():
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    url = f"http://localhost:{PORT}/index.html"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"MLP Demo — Team10")
    print(f"Running at {url}")
    print("Close this window to stop the server.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
