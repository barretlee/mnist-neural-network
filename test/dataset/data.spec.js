const { expect } = require('chai');
const fs = require('fs');
const path = require('path');
const { loadMNIST } = require('../../src/dataset');

describe('MNIST Data Loader', function () {
  it('should load the first image and save to test/tmp/first-image.png', function () {
    const mnist = loadMNIST();
    expect(mnist.train.images.length).to.be.greaterThan(0, 'No images loaded');
    const img = mnist.train.images[0];
    expect(img.length).to.equal(28 * 28, 'Image size mismatch');

    // Convert to PNG and save
    const { PNG } = require('pngjs');
    const png = new PNG({ width: 28, height: 28 });
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const idx = (y * 28 + x);
        const val = Math.round(img[idx] * 255);
        const ptr = (y * 28 + x) << 2;
        png.data[ptr] = val;      // R
        png.data[ptr + 1] = val; // G
        png.data[ptr + 2] = val; // B
        png.data[ptr + 3] = 255; // A
      }
    }
    const outPath = path.join(__dirname, '../tmp/first-image.png');
    fs.writeFileSync(outPath, PNG.sync.write(png));
    expect(fs.existsSync(outPath)).to.be.true;
  });
});