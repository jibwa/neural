import { JNetwork } from './lib/networks.mjs';

export default class XORTrainer {
  constructor(learningRate = 0.05) {
    Object.assign(this, {
      network: new JNetwork({
        input: 2,
        hidden: [2],
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
    const { trainingSet, network, learningRate } = this;
    /* let bigSet = [];
    const shuffleArray = arr => arr.sort(() => Math.random() - 0.5);
    Array(1).fill().forEach( () => {
     bigSet = bigSet.concat(shuffleArray(trainingSet.slice()))
    });
    */

    for (let i = 0; i < iterations; i += 1) {
      network.clearConnectionSums();
      trainingSet.forEach(([input, output]) => {
        network.activate(input);
        network.propagateSignal(output);
        network.weightSumConnections();
      });
      network.updateWeights(learningRate);
    }
    if (reportCallback) {
      reportCallback();
    }
  }
}
