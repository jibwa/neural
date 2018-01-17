import { JNetwork } from './lib/networks.mjs';

export default class XORTrainer {
  constructor(learningRate = 0.005, jsonData) {
    Object.assign(this, {
      network: jsonData ? new JNetwork(null, jsonData) : new JNetwork({
        input: 2,
        hidden: [3],
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
      trainingSet.forEach(([input, output]) => {
        network.activate(input);
        if (reportCallback) {
          reportCallback();
        }
        network.propagate(output);
        if (reportCallback) {
          reportCallback();
        }
      });
    }
  }
}
