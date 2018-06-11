const predict = (network, dataSet) => {
  network.scaleDropoutPForPrediction();
  const preds = dataSet.map(([features]) => network.activate(features, true));
  network.scaleDropoutPForPrediction(true);
  return preds;
}
export { predict };
