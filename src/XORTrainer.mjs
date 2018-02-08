import { JNetwork } from './lib/networks.mjs';

export default class XORTrainer {
  constructor(learningRate = 0.05, jsonData) {
    Object.assign(this, {
      network: jsonData ? new JNetwork(null, jsonData) : new JNetwork({
        input: 2,
        hidden: [300],
        output: 1,
        learningRate
      }),
      trainingSet: [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
      ]
    });
  }

  train(iterations, reportCallback) {
    const { trainingSet, network } = this;
    for (let i = 0; i < iterations; i += 1) {
      network.clearConnectionSums();
      trainingSet.forEach(([input, output]) => {
        network.activate(input);
        network.propagateSignal(output);
        network.weightSumConnections();
      });
      network.updateWeights(trainingSet.length);
    }
    if (reportCallback) {
      reportCallback();
    }
  }
}
