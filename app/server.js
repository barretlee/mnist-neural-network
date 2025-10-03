const http = require('http');
const fs = require('fs');
const path = require('path');
const { loadNetworkFromBuild } = require('../src/networkLoader');
const { createCanvas, loadImage } = require('canvas');

// 端口
const PORT = 8080;

// 识别图片数据
async function predictDigit(imageDataURL) {
  const net = loadNetworkFromBuild(true);
  if (!net) return { error: 'No trained model found.' };

  try {
    // 从 DataURL 加载图片
    const img = await loadImage(imageDataURL);
    const canvas = createCanvas(28, 28);
    const ctx = canvas.getContext('2d');
    
    // 绘制并缩放图片到 28x28
    ctx.drawImage(img, 0, 0, 28, 28);
    const imgData = ctx.getImageData(0, 0, 28, 28).data;
    
    // 转为灰度数组
    const input = [];
    for (let i = 0; i < imgData.length; i += 4) {
      // 简单平均灰度，考虑 alpha 通道
      input.push((imgData[i] + imgData[i+1] + imgData[i+2]) / 3 + (255 - imgData[i+3]));
    }
    
    // 归一化
    const normalized = input.map(v => Math.max(0, Math.min(1, v / 255)));
    
    let { output } = net.forward(normalized);
    output = output.map(v => v.toFixed(3) * 1000);
    return { prediction: pred, output };
  } catch (e) {
    return { error: 'Failed to process image' };
  }
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
    req.on('end', async () => {
      try {
        const { image } = JSON.parse(body);
        const result = await predictDigit(image);
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
