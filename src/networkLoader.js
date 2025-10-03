const { loadMNIST } = require('./dataset');
const fs = require('fs');
const path = require('path');
const NeuralNetwork = require('./network');

/**
 * 加载 build 目录下的神经网络模型（权重和配置）
 */
function loadNetworkFromBuild() {
  const buildDir = path.join(__dirname, '../build');
  const modelPath = path.join(buildDir, 'model.json');
  const configPath = path.join(buildDir, 'config.json');

  if (fs.existsSync(modelPath) && fs.existsSync(configPath)) {
    const model = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    const net = new NeuralNetwork(config.inputSize, config.hiddenSize, config.outputSize);
    net.import(model);
    return net;
  }
  return null;
}

module.exports = { loadNetworkFromBuild };
