import { JNetwork } from './lib/networks.mjs';
import { crossEntropy } from './lib/lossFunctions.mjs';
import { predict } from './lib/netHelpers.mjs';

/**
 * Shuffles array in place.
 * @param {Array} a items An array containing the items.
 */
function shuffle(a) {
  let j;
  let x;
  let i;
  for (i = a.length - 1; i > 0; i -= 1) {
    j = Math.floor(Math.random() * (i + 1));
    x = a[i];
    a[i] = a[j];
    a[j] = x;
  }
}

export default class Trainer {
  constructor({ layers, examples, trainConfig }) {
    const network = new JNetwork(layers);
    Object.assign(this, {
      totalIterations: 0,
      stopOrdered: false,
      trainConfig: {
        learningRate: 0.1,
        lossFn: crossEntropy,
        targetLoss: null,
        regularize: {
          level: 1,
          lambda: 0
        },
        ...trainConfig
      },
      network,
      info: {
        connectionSums: []
      },
      examples: {
        testSet: [],
        trainSet: [],
        ...examples
      },
      lastCost: 1,
      ...layers
    });
  }

  getData() {
    const {
      dataFn,
      examples,
      trainConfig: {
        trainTestRatio = 1,
        trainNum,
        randomizeSplit
      }
    } = this;

    return dataFn().then((results) => {
      if (examples.trainSet.length === 0) {
        examples.trainSet = results;
      } else {
        return true;
      }
      const { trainSet, testSet } = examples;
      let ratio = 1;
      if (trainNum > 0) {
        while (trainSet.length > trainNum) {
          let testExample;
          if (randomizeSplit) {
            const trainIndex = Math.floor(Math.random() * parseFloat(trainSet.length));
            [testExample] = trainSet.splice(trainIndex, 1);
          } else {
            testExample = trainSet.pop();
          }
          testSet.push(testExample);
        }
      } else {
        while (trainTestRatio < ratio) {
          let testExample;
          if (randomizeSplit) {
            const trainIndex = Math.floor(Math.random() * parseFloat(trainSet.length));
            [testExample] = trainSet.splice(trainIndex, 1);
          } else {
            testExample = trainSet.pop();
          }
          testSet.push(testExample);
          ratio = parseFloat(trainSet.length) / parseFloat(testSet.length);
        }
      }
      // self.hasData = true;
      return true;
    });
  }
  runFullStep(trainSet) {
    const { network } = this;
    network.clearConnectionSums();
    trainSet.forEach(([features, labels]) => {
      network.activate(features);
      network.calculateSignal(labels);
      network.updateConnectionSums();
    });
  }

  checkGradient() {
    const {
      network,
      examples: { trainSet },
      trainConfig: {
        epsilon = 0.001,
        regularize,
        lossFn
      }
    } = this;
    const connections = network.getConnections();

    const calcLoss = () => {
      const zs = predict(network, trainSet);
      return lossFn(zs, trainSet, regularize, connections);
    };
    this.runFullStep(trainSet);


    return connections.map((connection) => {
      connection.w += epsilon;
      const plusEpsilon = calcLoss();
      connection.w -= (2 * epsilon);
      const minusEpsilon = calcLoss();
      connection.w += epsilon;
      const { CID, errorSum } = connection;
      const numericGradient = (plusEpsilon - minusEpsilon) / (2.0 * epsilon);
      const analyticGradient = errorSum / parseFloat(trainSet.length);
      return {
        CID,
        numericGradient,
        analyticGradient
      };
    });
  }
  quickReport(i) {
    const {
      network,
      examples: { trainSet, testSet },
      trainConfig: {
        regularize,
        lossFn
      }
    } = this;
    const connections = network.getConnections();

    // simple round to 5 function
    const r5 = fl => parseFloat(fl.toFixed(5));

    const trainPreds = predict(network, trainSet);
    const trainLoss = lossFn(trainPreds, trainSet, regularize, connections);
    this.deltaLoss = this.lastCost - trainLoss;
    this.lastCost = trainLoss;
    if (testSet.length > 0) {
      const testPreds = predict(network, testSet);
      const testLoss = lossFn(testPreds, testSet, regularize, connections);
      console.log(`Loss (tr/te): ${trainLoss} / ${testLoss}   I: ${i}  dCost: ${this.deltaLoss}`);
      return testLoss;
    }
    console.log(`Loss: ${r5(trainLoss)} Delta: ${this.deltaLoss} I: ${i}`);
    return trainLoss;
  }
  train(iterations = 1000) {
    const {
      info,
      network,
      examples: {
        trainSet
      },
      trainConfig: {
        learningRate,
        regularize,
        shuffleAfterEpoch,
        targetLoss
      }
    } = this;

    const batchSize = this.trainConfig.batchSize || trainSet.length;

    // TODO REMOVE REMOVE
    console.log(`Training ${trainSet.length} examples in batches of size: ${batchSize} iterations: ${iterations}`);
    let i = 0;
    // const beginning = Date.now();
    while (i < iterations) {
      if (shuffleAfterEpoch === true) {
        shuffle(trainSet);
      }
      let batchIndex = 0;
      while (batchIndex < trainSet.length) {
        const batchExamples = trainSet.slice(batchIndex, batchSize + batchIndex);
        network.dropout();
        this.runFullStep(batchExamples);
        if (i === iterations.length - 1) {
          info.connectionSums = network.getConnectionSums();
        }
        network.updateWeights(learningRate, batchSize, regularize);
        network.restoreDropout();
        batchIndex += batchSize;
      }
      i += 1;
      this.totalIterations += 1
      if (i % 25 === 0) {
        // const reportStart = Date.now();
        const loss = this.quickReport(i);
        if (targetLoss && targetLoss > loss) {
          console.log('Target Loss Reached : ', targetLoss);
          i = iterations;
        }
        if (!this.stopOrdered && this.deltaLoss < 0.0) {
          i = iterations;
          this.stopOrdered = true;
        }
      }
    }
    console.log('TRAINING COMPLETE');
    return 'DONE';
  }

  pred() {
    const {
      network,
      examples: {
        trainSet
      }
    } = this;
    return trainSet.map(([feature, label]) => [feature, label, network.activate(feature)]);
  }
}
