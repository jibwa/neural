const netAverage = (fn, negate) =>
  (predictions, dataSet) => {
    const n = negate ? -1 : 1;
    return n * dataSet.reduce((acc, [input, output], index) =>
      acc + fn(output[0], predictions[index][0])
    , 0.0) / parseFloat(dataSet.length);
  }

const simpleCost = netAverage((y, z) =>
  y * Math.log(z) + (1-y) * Math.log(1-z), true);

const cost = (predictions, dataset, regularize, weights) => {
  const preReg = simpleCost(predictions, dataset)
  if (regularize == 0) {
    return preReg;
  }
  const weightSqSum = weights.reduce((acc, { w }) => acc + w * w, 0);
  const regC = regularize / ( 2 * predictions.length);
  return preReg + (regC * weightSqSum);
}

const bool = netAverage((y, z) =>
  Math.abs(y - Math.round(z)));

const meanSquare = netAverage((y, z) =>
  Math.pow(y - z, 2));

export { cost, bool, meanSquare };
