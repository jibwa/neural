import { JNetwork } from './lib/networks.mjs';

export default class XORTrainer {
  constructor(learningRate = 0.5) {
    Object.assign(this, {
      network: new JNetwork({
        input: { numNeurons: 5 },
        hidden: [{ numNeurons: 2 }],
        output: { numNeurons: 1, skipBias: true },
        learningRate
      }),
      trainingSet: [
        [[1,0,0,0,1],[.25]],
        [[1,0,0,1,0],[.25]],
        [[1,0,1,0,0],[.25]],
        [[1,1,0,0,0],[.25]],
        [[1,0,0,1,1],[.5]],
        [[1,0,1,0,1],[.5]],
        [[1,0,1,1,0],[.5]],
        [[1,1,0,1,0],[.5]],
        [[1,1,1,0,0],[.5]],
        [[1,1,0,0,1],[.5]],
        [[1,0,1,1,1],[.75]],
        [[1,1,0,1,1],[.75]],
        [[1,1,1,0,1],[.75]],
        [[1,1,1,1,0],[.75]]
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
