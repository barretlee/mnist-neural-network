
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const DATASET_DIR = path.join(__dirname, '../dataset');

function readGzFileSync(filePath) {
  const buffer = fs.readFileSync(filePath);
  return zlib.gunzipSync(buffer);
}

function parseImages(buffer) {
  // MNIST image file format:
  // [offset] [type]          [value]          [description]
  // 0000     32 bit integer  0x00000803(2051) magic number
  // 0004     32 bit integer  60000            number of images
  // 0008     32 bit integer  28               number of rows
  // 0012     32 bit integer  28               number of columns
  // 0016     unsigned byte   ??               pixel
  const magic = buffer.readUInt32BE(0);
  if (magic !== 2051) throw new Error('Invalid MNIST image file magic number');
  const numImages = buffer.readUInt32BE(4);
  const numRows = buffer.readUInt32BE(8);
  const numCols = buffer.readUInt32BE(12);
  const images = [];
  let offset = 16;
  for (let i = 0; i < numImages; i++) {
    const img = [];
    for (let j = 0; j < numRows * numCols; j++) {
      img.push(buffer[offset++] / 255);
    }
    images.push(img);
  }
  return images;
}

function parseLabels(buffer) {
  // MNIST label file format:
  // [offset] [type]          [value]          [description]
  // 0000     32 bit integer  0x00000801(2049) magic number
  // 0004     32 bit integer  60000            number of items
  // 0008     unsigned byte   ??               label
  const magic = buffer.readUInt32BE(0);
  if (magic !== 2049) throw new Error('Invalid MNIST label file magic number');
  const numLabels = buffer.readUInt32BE(4);
  const labels = [];
  let offset = 8;
  for (let i = 0; i < numLabels; i++) {
    labels.push(buffer[offset++]);
  }
  return labels;
}

function oneHot(label, numClasses = 10) {
  const arr = Array(numClasses).fill(0);
  arr[label] = 1;
  return arr;
}

function loadMNIST() {
  // Load training and test sets
  const trainImagesPath = path.join(DATASET_DIR, 'train-images-idx3-ubyte.gz');
  const trainLabelsPath = path.join(DATASET_DIR, 'train-labels-idx1-ubyte.gz');
  const testImagesPath = path.join(DATASET_DIR, 't10k-images-idx3-ubyte.gz');
  const testLabelsPath = path.join(DATASET_DIR, 't10k-labels-idx1-ubyte.gz');

  const trainImages = parseImages(readGzFileSync(trainImagesPath));
  const trainLabels = parseLabels(readGzFileSync(trainLabelsPath));
  const testImages = parseImages(readGzFileSync(testImagesPath));
  const testLabels = parseLabels(readGzFileSync(testLabelsPath));

  return {
    train: {
      images: trainImages,
      labels: trainLabels,
    },
    test: {
      images: testImages,
      labels: testLabels,
    },
  };
}

module.exports = {
  loadMNIST,
  oneHot,
};
