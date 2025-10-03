const http = require('http');
const fs = require('fs');
const path = require('path');
const { loadNetworkFromBuild } = require('../src/networkLoader');

// 端口
const PORT = 8080;

// 识别图片数据
function predictDigit(imageData) {
  const net = loadNetworkFromBuild(true);
  if (!net) return { error: 'No trained model found.' };

  // 压缩图片到 28x28
  function resizeTo28x28(data, srcWidth, srcHeight) {
    const dst = [];
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        // 最近邻采样
        const srcX = Math.floor(x * srcWidth / 28);
        const srcY = Math.floor(y * srcHeight / 28);
        dst.push(data[srcY * srcWidth + srcX]);
      }
    }
    return dst;
  }

  let input = imageData;
  // 如果不是 784 长度，尝试压缩
  if (input.length !== 784) {
    const side = Math.sqrt(input.length);
    if (Number.isInteger(side)) {
      input = resizeTo28x28(input, side, side);
    } else {
      // 长度异常，补零
      input = input.concat(Array(784 - input.length).fill(0)).slice(0, 784);
    }
  }

  input = input.map(v => Math.max(0, Math.min(1, v / 255)));

  const { output } = net.forward(input);
  const pred = output.indexOf(Math.max(...output));
  return { prediction: pred, output };
}

// 简单静态文件服务
function serveStatic(res, filePath, contentType) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not found');
    } else {
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(data);
    }
  });
}

const server = http.createServer((req, res) => {
  if (req.method === 'GET' && (req.url === '/' || req.url === '/canvas.html')) {
    serveStatic(res, path.join(__dirname, 'canvas.html'), 'text/html');
  } else if (req.method === 'GET' && req.url.startsWith('/')) {
    // 静态资源
    const ext = path.extname(req.url);
    let type = 'text/plain';
    if (ext === '.js') type = 'application/javascript';
    if (ext === '.css') type = 'text/css';
    serveStatic(res, path.join(__dirname, req.url), type);
  } else if (req.method === 'POST' && req.url === '/predict') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const { image } = JSON.parse(body);
        const result = predictDigit(image);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (e) {
        res.writeHead(400);
        res.end('Invalid request');
      }
    });
  } else {
    res.writeHead(404);
    res.end('Not found');
  }
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
});
