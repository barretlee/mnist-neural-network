const { loadMNIST } = require('./dataset');
const fs = require('fs');
const path = require('path');
const NeuralNetwork = require('./network');

/**
 * 从 build 目录加载已训练好的神经网络模型
 * 如果模型和配置文件存在，则装配并返回 NeuralNetwork 实例
 * 否则返回 null
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
