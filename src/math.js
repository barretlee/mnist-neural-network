// 数学工具函数集合，用于神经网络的激活函数和权重初始化

/**
 * Sigmoid 激活函数
 * 将输入值映射到 (0, 1) 区间，常用于神经网络的非线性变换。
 * 原理：sigmoid(x) = 1 / (1 + exp(-x))
 * 优点：输出平滑，易于求导
 */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Sigmoid 的导数
 * 用于反向传播时计算梯度。
 * 原理：sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
 */
function sigmoidDerivative(x) {
  const s = sigmoid(x);
  return s * (1 - s);
}

/**
 * 随机初始化权重
 * 神经网络权重通常初始化为较小的随机值，以打破对称性。
 * @param {number} size 数组长度
 * @returns {Array<number>}
 */
function randomArray(size) {
  // 权重初始化为 [-1, 1] 区间的随机数
  return Array.from({ length: size }, () => Math.random() * 2 - 1);
}

module.exports = {
  sigmoid,
  sigmoidDerivative,
  randomArray,
};