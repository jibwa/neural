const netAverage = (fn) =>
  (predictions, dataSet) =>
    dataSet.reduce((acc, [input, output], index) =>
      acc + fn(output[0], predictions[index][0])
    , 0.0) / parseFloat(dataSet.length);

const cost = netAverage((y, z) =>
  y * Math.log(z) + (1-y) * Math.log(1-z));

const bool = netAverage((y, z) =>
  Math.abs(y - Math.round(z)));

const meanSquare = netAverage((y, z) =>
  Math.pow(y - z, 2));

export { cost, bool, meanSquare };
