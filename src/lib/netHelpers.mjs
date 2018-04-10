const predict = (network, dataSet) =>
  dataSet.map(([input]) => network.activate(input, true));

export { predict };
