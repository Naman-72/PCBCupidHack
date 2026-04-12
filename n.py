from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import socket

HOST = "0.0.0.0"
PORT = 8000


class MyHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # If root is requested, serve index.html
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def main():
    server = ThreadingHTTPServer((HOST, PORT), MyHandler)
    local_ip = get_local_ip()

    print(f"Open on mobile: http://{local_ip}:{PORT}")
    print("Serving index.html as default page")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()