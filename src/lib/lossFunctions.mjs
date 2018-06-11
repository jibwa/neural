const netAverage = (fn, negate) =>
  (predictions, dataSet) => {
    const n = negate ? -1 : 1;
    return n * dataSet.reduce((acc, [feature, label], index) =>
      acc + fn(label, predictions[index])
    , 0.0) / parseFloat(dataSet.length);
  }

const simpleCrossEntropy = netAverage((yArr, zArr) => {
  const y = yArr[0];
  const p = zArr[0];
  return -y * Math.log(p) + (1-y) * Math.log(1-p);
});

const meanSquared = netAverage((y, p) => {
  return Math.pow(p - y, 2);
});

const crossEntropy = (predictions, dataset, { level, lambda }, weights) => {

  const preReg = simpleCrossEntropy(predictions, dataset)

  // no regularization applied so simple cost will do
  if (!(lambda > 0)) {
    return preReg;
  }
  console.log('SHOULD NEVER GET HERE');

  // apply regularization to cost
  const pLen = parseFloat(predictions.length);
  let weightSqSum;
  let regC;
  if (level === 1) {
    weightSqSum = weights.reduce((acc, { w }) => acc + Math.abs(w), 0);
    regC = lambda / pLen
  } else if (level === 2) {
    weightSqSum = weights.reduce((acc, { w }) => acc + Math.pow(w, 2), 0);
    regC = lambda / (2.0 * pLen)
  }
  return preReg + (regC * weightSqSum);
/*
  const weightSqSum = weights.reduce((acc, { w }) => acc + w * w, 0);
  const regC = regularize / ( 2 * predictions.length);
  return preReg + (regC * weightSqSum);
  */
}

const simpleSoftmaxXECost = netAverage((yArray, zArray) => {
  const index = yArray.indexOf(1);
  return -yArray[index] * Math.log(zArray[index]);
});

const softmaxXECost = (predictions, dataset, { level, lambda }, weights) => {
  const preReg = simpleSoftmaxXECost(predictions, dataset)
  // regularization
  if (lambda === 0) {
    return preReg;
  }
  const pLen = parseFloat(predictions.length);
  let weightSqSum;
  let regC;
  if (level === 1) {
    weightSqSum = weights.reduce((acc, { w }) => acc + Math.abs(w), 0);
    regC = lambda / pLen
  } else if (level === 2) {
    weightSqSum = weights.reduce((acc, { w }) => acc + Math.pow(w, 2), 0);
    regC = lambda / (2.0 * pLen)
  }
  return preReg + (regC * weightSqSum);
}

const softmaxBool = netAverage((yArray, zArray) => {
  const index = yArray.indexOf(1);
  const indexOfMaxValue = zArray.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
  return index === indexOfMaxValue ? 0 : 1;
})


const meanAbsolute = netAverage((y, p) =>
  Math.abs(y - p));
export { crossEntropy,  meanSquared, softmaxXECost, softmaxBool };
