const fs = require('fs');
const path = require('path');
const NeuralNetwork = require('./network');
const { loadMNIST } = require('./dataset');

// 创建 build 目录（用于保存模型和训练结果）
const buildDir = path.join(__dirname, '../build');
if (!fs.existsSync(buildDir)) fs.mkdirSync(buildDir);

// 加载 MNIST 数据集
// MNIST 是一个手写数字图片数据集，常用于神经网络入门
const mnist = loadMNIST();
const trainImages = mnist.train.images; // 使用全部训练样本
const trainLabels = mnist.train.labels;
const testImages = mnist.test.images;
const testLabels = mnist.test.labels;

// 网络结构配置
const config = {
  inputSize: 28 * 28, // MNIST 图片为 28x28 像素
  hiddenSize: 64,     // 隐藏层节点数，可调整
  outputSize: 10,     // 输出层节点数，对应数字 0-9
  learningRate: 0.1,  // 学习率，控制每次权重调整幅度
  epochs: 10,         // 训练轮数
};

// 初始化神经网络
const net = new NeuralNetwork(config.inputSize, config.hiddenSize, config.outputSize);

/**
 * 将标签转为 one-hot 编码
 * 原理：将数字标签转为长度为 10 的数组，只有对应位置为 1，其余为 0
 * 例如标签 3 -> [0,0,0,1,0,0,0,0,0,0]
 */
function oneHot(label, numClasses = 10) {
  const arr = Array(numClasses).fill(0);
  arr[label] = 1;
  return arr;
}

// 训练过程
// 原理：多轮遍历训练集，每个样本都进行前向和反向传播，不断优化权重
for (let epoch = 0; epoch < config.epochs; epoch++) {
  let totalLoss = 0;
  for (let i = 0; i < trainImages.length; i++) {
    const input = trainImages[i];
    const target = oneHot(trainLabels[i]);
    net.train(input, target, config.learningRate);

    // 计算损失（绝对误差均值）
    // 用于衡量预测输出与真实标签的差距
    const { output } = net.forward(input);
    const loss = target.reduce((sum, t, j) => sum + Math.abs(t - output[j]), 0) / config.outputSize;
    totalLoss += loss;
  }
  const avgLoss = totalLoss / trainImages.length;
  console.log(`Epoch ${epoch + 1}, Avg Loss: ${avgLoss.toFixed(4)}`);
}

// 保存模型参数和配置到 build 目录
fs.writeFileSync(path.join(buildDir, 'model.json'), JSON.stringify(net.export(), null, 2));
fs.writeFileSync(path.join(buildDir, 'config.json'), JSON.stringify(config, null, 2));

// 测试网络（输出前 5 个样本预测结果）
console.log('\n测试网络输出:');
for (let i = 0; i < 5; i++) {
  const input = testImages[i];
  const label = testLabels[i];
  const { output } = net.forward(input);
  const pred = output.indexOf(Math.max(...output));
  // 输出预测结果和真实标签
  console.log(`样本 ${i + 1}: 真实标签=${label}, 预测标签=${pred}, 输出=${output.map(o => o.toFixed(2))}`);
}
