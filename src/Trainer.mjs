import { JNetwork } from './lib/networks.mjs';
import { cost, bool } from './lib/lossFunctions.mjs';
import { predict } from './lib/netHelpers.mjs';

/**
 * Shuffles array in place.
 * @param {Array} a items An array containing the items.
 */
function shuffle(a) {
    var j, x, i;
    for (i = a.length - 1; i > 0; i--) {
        j = Math.floor(Math.random() * (i + 1));
        x = a[i];
        a[i] = a[j];
        a[j] = x;
    }
}

export default class Trainer {
  constructor(def) {
    const network = new JNetwork(def.layers);
    Object.assign(this, {
      ...def,
      network,
      info: {
        connectionSums: []
      },
      examples: {
        testSet: [],
        trainSet: []
      }
    });
  }

  getData() {
    const {
      dataFn,
      examples,
      trainConfig: {
        trainTestRatio = 1,
        randomizeSplit
      }
    } = this;

    return dataFn().then(results => {
      if (examples.trainSet.length === 0) {
        examples.trainSet = results;
      } else {
        return true;
      }
      const { trainSet, testSet } = examples;
      let ratio = 1;
      while (trainTestRatio < ratio) {
        let testExample;
        if (randomizeSplit) {
          const trainIndex = Math.floor(Math.random() * parseFloat(trainSet.length));
          testExample = trainSet.splice(trainIndex,1)[0];
        } else {
          testExample = trainSet.pop();
        }
        testSet.push(testExample);
        ratio = parseFloat(trainSet.length) / parseFloat(testSet.length);
      }
      // self.hasData = true;
      return true;
    });
  }
  runFullStep() {
    const { network, examples: { trainSet } } = this;
    trainSet.forEach(([input, output]) => {
      network.activate(input);
      network.calculateSignal(output);
      network.updateConnectionSums();
    });
  }

  calcGradient() {
    const {
      network,
      examples: { trainSet },
      trainConfig: { epsilon = 0.001 }
    } = this;
    const calcLoss = () => {
      const zs = predict(network, trainSet);
      return cost(zs, trainSet);
    };
    this.runFullStep();

    const connections = network.getConnections();

    return connections.map((connection) => {
      connection.w = connection.w + epsilon;
      const plusEpsilon = calcLoss();
      connection.w = connection.w - (2 * epsilon);
      const minusEpsilon = calcLoss();
      connection.w = connection.w + epsilon;
      const { CID, errorSum } = connection;
      return {
        CID,
        bpDelta: errorSum / parseFloat(trainSet.length),
        epDelta: (plusEpsilon - minusEpsilon) / (2.0 * epsilon)
      }
    });
  }
  quickReport(i) {
    const { network, examples: { trainSet, testSet } } = this;
    const r5 = (fl) => parseFloat(fl.toFixed(5));
    const trainPreds = predict(network, trainSet);
    const trainCost = r5(cost(trainPreds, trainSet));
    const trainBool = r5(bool(trainPreds, trainSet));
    if (testSet.length > 0) {
      const testPreds = predict(network, testSet);
      const testCost = r5(cost(testPreds, testSet))
      const testBool = r5(bool(testPreds, testSet));
      console.log(`Costs (tr/te): ${trainCost} / ${testCost}   Bool: ${trainBool} / ${testBool}  I: ${i}`);
      return testBool;
    } else {

      console.log(`Cost: ${trainCost}  Bool: ${trainBool}  I: ${i}`);
    }
  }
  train(iterations = 1000, seed = 1) {
    const {
      examples,
      info,
      network,
      examples: {
        trainSet,
        testSet
      },
      trainConfig: {
        learningRate,
        regularize,
        batchSize,
        shuffleAfterEpoch
      }
    } = this;

    // TODO REMOVE REMOVE
    console.log(`Training ${trainSet.length} examples in batches of size: ${batchSize} iterations: ${iterations}`);
    this.quickReport(0);
    let i = 0;
    const dinfo = {
      activation: 0,
      backProp: 0,
      reporting: 0,
      updating: 0,
      total: 0
    }
    // const beginning = Date.now();
    while (i < iterations) {
      if (shuffleAfterEpoch === true) {
        shuffle(trainSet);
      }
      let batchIndex = 0;
      while (batchIndex < trainSet.length) {
        const batchExamples = trainSet.slice(batchIndex, batchSize + batchIndex);
        network.dropout();
        batchExamples.forEach(([input, output]) => {
          // const actStart = Date.now();
          network.activate(input);
          // const actEnd = Date.now()
          //dinfo.activation += (actEnd - actStart);
          network.calculateSignal(output);
          network.updateConnectionSums();
          // dinfo.backProp += Date.now() - actEnd;
        });
        if (i === iterations.length - 1) {
          info.connectionSums = network.getConnectionSums();
        }
        // const updateStart = Date.now();
        network.updateWeights(learningRate, batchSize, regularize);
        network.restoreDropout();
        // dinfo.updating += Date.now() - updateStart;
        // network.clearConnectionSums();
        batchIndex += batchSize;
      }
      i += 1;
      if (i % 100 === 0) {
        // const reportStart = Date.now();
        const testBool = this.quickReport(i);
        if (testBool < .22) {
          i = iterations;
        }
        // dinfo.reporting = Date.now() - reportStart;
        // console.log(dinfo);
      }

    }
    console.log('Training Complete');

    return 'DONE';
  }

}


