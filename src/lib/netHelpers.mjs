const predict = (network, dataSet) => {
  network.scaleDropoutPForPrediction();
  const preds = dataSet.map(([input]) => network.activate(input, true));
  network.scaleDropoutPForPrediction(true);
  return preds;
}
export { predict };
