const { sigmoid, sigmoidDerivative, randomArray } = require('./math');

/**
 * 简单的全连接前馈神经网络（单隐藏层）
 * 原理说明：
 * - 输入层、隐藏层、输出层均为全连接
 * - 前向传播：输入经过权重和激活函数，逐层输出
 * - 反向传播：通过损失函数计算误差，反向调整权重
 */
class NeuralNetwork {
  /**
   * 构造函数，初始化权重和偏置
   * @param {number} inputSize 输入层节点数
   * @param {number} hiddenSize 隐藏层节点数
   * @param {number} outputSize 输出层节点数
   */
  constructor(inputSize, hiddenSize, outputSize) {
    // 权重矩阵：输入层到隐藏层
    this.weightsIH = Array.from({ length: hiddenSize }, () => randomArray(inputSize));
    // 隐藏层偏置
    this.biasH = randomArray(hiddenSize);
    // 权重矩阵：隐藏层到输出层
    this.weightsHO = Array.from({ length: outputSize }, () => randomArray(hiddenSize));
    // 输出层偏置
    this.biasO = randomArray(outputSize);

    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
  }

  /**
   * 前向传播
   * 原理：每一层的输出 = 激活函数(权重 * 输入 + 偏置)
   * @param {Array<number>} input 输入向量
   * @returns {Object} { hidden, output }
   */
  forward(input) {
    // 输入 -> 隐藏层
    // 对每个隐藏节点，计算加权和并激活
    const hidden = this.weightsIH.map((w, i) =>
      sigmoid(w.reduce((sum, wi, j) => sum + wi * input[j], this.biasH[i]))
    );
    // 隐藏层 -> 输出层
    // 对每个输出节点，计算加权和并激活
    const output = this.weightsHO.map((w, i) =>
      sigmoid(w.reduce((sum, wi, j) => sum + wi * hidden[j], this.biasO[i]))
    );
    return { hidden, output };
  }

  /**
   * 反向传播与权重更新
   * 原理：
   * - 计算输出误差
   * - 通过激活函数的导数计算梯度
   * - 按梯度方向调整权重和偏置
   * @param {Array<number>} input 输入向量
   * @param {Array<number>} target 目标输出（如 one-hot 编码）
   * @param {number} lr 学习率
   */
  train(input, target, lr = 0.1) {
    // 前向传播
    const { hidden, output } = this.forward(input);

    // 输出层误差 = 目标值 - 实际输出
    const outputErrors = output.map((o, i) => target[i] - o);

    // 输出层梯度 = 误差 * 激活函数导数
    // 这里的 sigmoidDerivative 传入的是输出值（已激活），理论上应传入加权和，但对 sigmoid 可直接用输出值
    const outputGrad = output.map((o, i) => outputErrors[i] * o * (1 - o));

    // 更新隐藏到输出层权重和偏置
    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        // 权重调整 = 学习率 * 梯度 * 上一层输出
        this.weightsHO[i][j] += lr * outputGrad[i] * hidden[j];
      }
      // 偏置调整
      this.biasO[i] += lr * outputGrad[i];
    }

    // 隐藏层误差 = 输出层梯度反向传播到隐藏层
    // 对每个隐藏节点，累加所有输出节点的梯度 * 权重
    const hiddenErrors = this.weightsHO[0].map((_, j) =>
      outputGrad.reduce((sum, og, k) => sum + og * this.weightsHO[k][j], 0)
    );

    // 隐藏层梯度 = 误差 * 激活函数导数
    const hiddenGrad = hidden.map((h, i) => hiddenErrors[i] * h * (1 - h));

    // 更新输入到隐藏层权重和偏置
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.inputSize; j++) {
        this.weightsIH[i][j] += lr * hiddenGrad[i] * input[j];
      }
      this.biasH[i] += lr * hiddenGrad[i];
    }
  }

  /**
   * 导出模型参数
   * 用于保存训练好的权重和偏置
   */
  export() {
    return {
      inputSize: this.inputSize,
      hiddenSize: this.hiddenSize,
      outputSize: this.outputSize,
      weightsIH: this.weightsIH,
      biasH: this.biasH,
      weightsHO: this.weightsHO,
      biasO: this.biasO,
    };
  }
}

module.exports = NeuralNetwork;
