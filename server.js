// Tiny static server for the browser version (no dependencies).
const http = require("http");
const fs = require("fs");
const path = require("path");

const root = __dirname;
const port = process.env.PORT || 8000;
const types = {
  ".html": "text/html", ".css": "text/css", ".js": "text/javascript",
  ".mjs": "text/javascript", ".png": "image/png", ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg", ".mp3": "audio/mpeg", ".wav": "audio/wav",
  ".task": "application/octet-stream", ".wasm": "application/wasm",
};

http.createServer((req, res) => {
  const urlPath = decodeURIComponent(req.url.split("?")[0]);
  const file = path.join(root, urlPath === "/" ? "index.html" : urlPath);
  if (!file.startsWith(root)) { res.writeHead(403); return res.end(); }
  fs.stat(file, (err, stat) => {
    if (err || !stat.isFile()) { res.writeHead(404); return res.end("Not found"); }
    res.writeHead(200, {
      "Content-Type": types[path.extname(file)] || "application/octet-stream",
      "Cache-Control": "no-store",   // edited assets/audio show up at once
    });
    if (req.method === "HEAD") return res.end();
    fs.createReadStream(file).pipe(res);
  });
}).listen(port, () => console.log(`Sharingan web: http://localhost:${port}`));
