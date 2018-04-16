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
      stopOrdered: false,
      ...def,
      trainConfig: {
        costFn: cost,
        boolFn: bool,
        regularize: {
          level: 1,
          lambda: 0,
        },
        // for binary and softmax
        skipOccurances: false,
        ...def.trainConfig
      },
      network,
      info: {
        connectionSums: []
      },
      examples: {
        testSet: [],
        trainSet: []
      },
      lastCost: 0
    });
  }

  getData() {
    const {
      dataFn,
      examples,
      trainConfig: {
        skipOccurances,
        trainTestRatio = 1,
        trainNum,
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
      if (trainNum > 0) {
        while (trainSet.length > trainNum) {
          let testExample;
          if (randomizeSplit) {
            const trainIndex = Math.floor(Math.random() * parseFloat(trainSet.length));
            testExample = trainSet.splice(trainIndex,1)[0];
          } else {
            testExample = trainSet.pop();
          }
          testSet.push(testExample);
        }
      }
      else {
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
      }
      if (skipOccurances) {
        const zeros = new Array(trainSet[0][1].length).fill(0);
        const occurances = trainSet.reduce((acc, [inputs, outputs]) => {
          const index = outputs.indexOf(1);
          if (index != -1) {
            acc[index] += 1;
          }
          return acc;
        }, zeros)
        if (zeros.length == 1) {
          // we are in binary classification mode
          // so se index 0 to be 0 scaler and keep index 1 as 1 scaler
          occurances.push(trainSet.length - occurances[0]);

        }
        const min = occurances.reduce((acc, occurance) => occurance < acc ? occurance : acc, trainSet.length)
        let occuranceSkips = occurances.map(n => min / parseFloat(n));
        console.log(`Skipping Occurances: ${occurances} => ${occuranceSkips}`);
        this.occuranceSkips = occuranceSkips;

        // self.hasData = true;
      }
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
      trainConfig: {
        epsilon = 0.0001,
        regularize,
        costFn
      }
    } = this;
    const connections = network.getConnections();

    const calcLoss = () => {
      const zs = predict(network, trainSet);
      return costFn(zs, trainSet, regularize, connections);
    };
    this.runFullStep();


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
    const {
      network,
      examples: { trainSet, testSet },
      trainConfig: { regularize, costFn, boolFn }
    } = this;
    const connections = network.getConnections();
    const r5 = (fl) => parseFloat(fl.toFixed(5));
    const trainPreds = predict(network, trainSet);
    const trainCost = costFn(trainPreds, trainSet, regularize, connections);
    this.deltaCost = this.lastCost - trainCost;
    this.lastCost = trainCost;
    const trainBool = r5(boolFn(trainPreds, trainSet));
    if (testSet.length > 0) {
      const testPreds = predict(network, testSet);
      const testCost = costFn(testPreds, testSet, regularize, connections);
      const testBool = r5(boolFn(testPreds, testSet));
      console.log(`Costs (tr/te): ${trainCost} / ${testCost}   Bool: ${trainBool} / ${testBool}  I: ${i}  dCost: ${this.deltaCost}`);
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
        skipOccurances,
        learningRate,
        regularize,
        batchSize,
        shuffleAfterEpoch
      },
      occuranceSkips
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
        let finalBatchSize = 0
        network.dropout();
        batchExamples.forEach(([input, output]) => {
          // HERE - what if instead of randomly having different batch sizes I randomly filled the slots
          if (false && skipOccurances) {
            let skipOdds = 0;
            if (output.length === 1) {
              skipOdds = occuranceSkips[output[0]];
            } else {
              skipOdds = occuranceSkips[output.indexOf(1)];
            }
            if (Math.random() > skipOdds) {
              return;
            }
          }
          finalBatchSize += 1;
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
        network.updateWeights(learningRate, finalBatchSize, regularize);
        network.restoreDropout();
        // dinfo.updating += Date.now() - updateStart;
        // network.clearConnectionSums();
        batchIndex += batchSize;
      }
      i += 1;
      if (i % 50 === 0) {
        // const reportStart = Date.now();
        const testBool = this.quickReport(i);
        if (testBool < .10) {
          i = iterations;
        }
        if (!this.stopOrdered && this.deltaCost < 0.0) {
          i = iterations;
          this.stopOrdered = true
          console.log('HERE IT IS');
        }
        // dinfo.reporting = Date.now() - reportStart;
        // console.log(dinfo);
      }

    }
    console.log('Training Complete');

    return 'DONE';
  }

}

