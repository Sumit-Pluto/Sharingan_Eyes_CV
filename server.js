// Tiny static server for local development (no dependencies).
// Production (Vercel) serves these same files as static assets —
// see vercel.json. Keep the two in sync: mime types, cache policy.
const http = require("http");
const fs = require("fs");
const path = require("path");

const root = __dirname;
const port = process.env.PORT || 8000;
const types = {
  ".html": "text/html", ".css": "text/css", ".js": "text/javascript",
  ".mjs": "text/javascript", ".json": "application/json",
  ".png": "image/png", ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg", ".svg": "image/svg+xml",
  ".mp3": "audio/mpeg", ".wav": "audio/wav", ".mp4": "video/mp4",
  ".webm": "video/webm", ".onnx": "application/octet-stream",
  ".task": "application/octet-stream", ".wasm": "application/wasm",
};

// Versioned-by-content in practice (replaced on each deploy), so media
// can be cached; markup and code stay fresh for local iteration.
function cacheControl(file) {
  if (/\.(png|jpe?g|svg|mp3|wav|mp4|webm|wasm|onnx|task)$/i.test(file)) {
    return "public, max-age=3600";
  }
  return "no-store";
}

http.createServer((req, res) => {
  const urlPath = decodeURIComponent(req.url.split("?")[0]);
  const file = path.join(root, urlPath === "/" ? "index.html" : urlPath);
  if (!file.startsWith(root)) { res.writeHead(403); return res.end(); }
  fs.stat(file, (err, stat) => {
    if (err || !stat.isFile()) { res.writeHead(404); return res.end("Not found"); }
    const headers = {
      "Content-Type": types[path.extname(file)] || "application/octet-stream",
      "Cache-Control": cacheControl(file),
      "Accept-Ranges": "bytes",
    };
    // Single-range requests: mobile Safari will not playseekable
    // <video> without 206 Partial Content support.
    const range = req.headers.range;
    if (range) {
      const m = /^bytes=(\d*)-(\d*)$/.exec(range);
      const size = stat.size;
      const start = m && m[1] ? parseInt(m[1], 10) : 0;
      const end = m && m[2] ? parseInt(m[2], 10) : size - 1;
      if (m && start < size && end < size && start <= end) {
        res.writeHead(206, {
          ...headers,
          "Content-Range": `bytes ${start}-${end}/${size}`,
          "Content-Length": end - start + 1,
        });
        if (req.method === "HEAD") return res.end();
        return fs.createReadStream(file, { start, end }).pipe(res);
      }
      res.writeHead(416, { ...headers, "Content-Range": `bytes */${size}` });
      return res.end();
    }
    res.writeHead(200, { ...headers, "Content-Length": stat.size });
    if (req.method === "HEAD") return res.end();
    fs.createReadStream(file).pipe(res);
  });
}).listen(port, () => console.log(`Sharingan web: http://localhost:${port}`));
