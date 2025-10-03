const fs = require('fs');
const path = require('path');
const { expect } = require('chai');
const { loadNetworkFromBuild } = require('../../src/networkLoader');
const NeuralNetwork = require('../../src/network');
const { loadMNIST } = require('../../src/dataset');

describe('networkLoader', () => {
  const buildDir = path.join(__dirname, '../../build');
  const modelPath = path.join(buildDir, 'model.json');
  const configPath = path.join(buildDir, 'config.json');

  beforeEach(() => {
    // 每个测试前都准备好模型和配置文件
    if (!fs.existsSync(buildDir)) fs.mkdirSync(buildDir);
    const config = { inputSize: 2, hiddenSize: 2, outputSize: 2 };
    const net = new NeuralNetwork(2, 2, 2);
    fs.writeFileSync(modelPath, JSON.stringify(net.export()));
    fs.writeFileSync(configPath, JSON.stringify(config));
  });

  afterEach(() => {
    // 每个测试后都清理文件和目录
    if (fs.existsSync(modelPath)) fs.unlinkSync(modelPath);
    if (fs.existsSync(configPath)) fs.unlinkSync(configPath);
    if (fs.existsSync(buildDir)) {
      // 递归删除 build 目录
      fs.rmSync(buildDir, { recursive: true, force: true });
    }
  });

  it('should load network from build directory', () => {
    const net = loadNetworkFromBuild();
    expect(net).to.be.an.instanceof(NeuralNetwork);
    expect(net.inputSize).to.equal(2);
    expect(net.hiddenSize).to.equal(2);
    expect(net.outputSize).to.equal(2);
  });

  it('should return null if files do not exist', () => {
    // 清理文件
    if (fs.existsSync(modelPath)) fs.unlinkSync(modelPath);
    if (fs.existsSync(configPath)) fs.unlinkSync(configPath);
    const net = loadNetworkFromBuild();
    expect(net).to.be.null;
    // 恢复文件
    const config = { inputSize: 2, hiddenSize: 2, outputSize: 2 };
    const tmpNet = new NeuralNetwork(2, 2, 2);
    fs.writeFileSync(modelPath, JSON.stringify(tmpNet.export()));
    fs.writeFileSync(configPath, JSON.stringify(config));
  });

  it('should predict on a real MNIST image', () => {
    // 重新写入真实模型和配置
    if (fs.existsSync(modelPath)) fs.unlinkSync(modelPath);
    if (fs.existsSync(configPath)) fs.unlinkSync(configPath);

    // 加载 MNIST 数据集，获取真实 inputSize
    const mnist = loadMNIST();
    const image = mnist.train.images[0];
    const label = mnist.train.labels[0];
    const inputSize = image.length;
    const config = { inputSize, hiddenSize: 2, outputSize: 10 };
    const net = new NeuralNetwork(inputSize, 2, 10);
    fs.writeFileSync(modelPath, JSON.stringify(net.export()));
    fs.writeFileSync(configPath, JSON.stringify(config));

    const loadedNet = loadNetworkFromBuild();
    expect(loadedNet).to.be.an.instanceof(NeuralNetwork);

    // 检查图片数据格式
    expect(image).to.be.an('array').with.lengthOf(loadedNet.inputSize);
    expect(label).to.be.a('number').within(0, 9);

    // 网络预测
    const { output } = loadedNet.forward(image);
    expect(output).to.be.an('array').with.lengthOf(loadedNet.outputSize);

    // 输出最大值索引作为预测标签
    const pred = output.indexOf(Math.max(...output));
    expect(pred).to.be.a('number').within(0, 9);
  });
});
