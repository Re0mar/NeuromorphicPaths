# The page's certificate

`demo_page.crt` and `demo_page.key` are the self-signed certificate and private key the web sink
serves the page with. The page is HTTPS because the browser's own video decoder, WebCodecs, is only
exposed on a secure origin, and the page is opened on a phone by the laptop's address, which plain
http never makes secure.

The key protects nothing. The page is on a local network either way, and anyone on that network
who could read this key could as easily watch the plain traffic the page used to send. It is
committed so every laptop serves the same certificate and each browser accepts the warning once.
GitHub's secret scanning will flag a private key in a public repository. That is expected, and
this file is the answer to it.

Each browser shows a warning the first time it opens `https://<laptop>:8765`. Advanced, then
proceed, once per browser per device. Remake the pair, from this directory, if it ever expires:

```
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -keyout demo_page.key -out demo_page.crt -subj "/CN=nav demo page" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```

From Git Bash, put `MSYS_NO_PATHCONV=1` in front, or the shell rewrites the subject as a path.
