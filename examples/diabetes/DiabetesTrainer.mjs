import { getRainmanData, postWarrigData } from '../helpers.mjs';
import { inputDefs, outputDefs } from './definitions.mjs';
import { softplus, sigmoid, logits, relu } from '../../neural/src/lib/activationFunctions.mjs';

import Trainer from '../../neural/src/Trainer.mjs';
const errorFunctions = {
  squareReduceMean: 'squareReduceMean',
  sparseSoftmaxCrossEntropy: 'sparseSoftmaxCrossEntropy'
}
export default class DiabetesTrainer extends Trainer {
  constructor() {
    super({
      dataFn: () => getRainmanData('/diabetes'),
      stopOrdered: false,
      layers: {
        input: {
          numNeurons: 8,
          weightMax: .1,
          weightMin: 0,
          skipBias: false,
          wBound: 25.0,
          dropout: 1
        },
        hidden: [ {
          numNeurons: 8,
          activationFunction: sigmoid,
          weightMax: .1,
          weightMin: 0,
          skipBias: false,
          wBound: 25.0,
          dropout:1
        },
          {
            numNeurons: 8,
            activationFunction: sigmoid,
            weightMax: .1,
            weightMin: 0,
            skipBias: false,
            wBound: 25.0,
            dropout: 1
          }],
        output: {
          numNeurons: 1,
          activationFunction: sigmoid,
          skipBias: true
        }
      },
      trainConfig: {
        learningRate: 1.0,
        regularize: {
          level: 2,
          lambda: 0.001
        },
        errorFunction: errorFunctions.squareReduceMean,
        //lossFunction: lossFunctions.sparseSoftmaxCrossEntropy,
        batchSize: 150,
        trainTestRatio: 1.0,
        trainNum: 150,
        randomizeSplit: false,
        shuffleAfterEpoch: false,
        epsilon: 0.001
      }
    });
  }

  outputResults() {
    const {
      examples : { trainSet, testSet },
      network
    } = this;
    const results = [];
    return {
      train: trainSet.reduce((acc, [input, output]) => {
        acc[output.toString()].push(network.activate(input)[0]);
        return acc;
      }, {'0': [], '1': []}),
      test: testSet.reduce((acc, [input, output]) => {
        acc[output.toString()].push(network.activate(input)[0]);
        return acc;
      }, {'0': [], '1': []})
    };
  }

}
