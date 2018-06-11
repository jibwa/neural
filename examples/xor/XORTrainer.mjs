import Trainer from '../../src/Trainer.mjs';
import { crossEntropy, meanSquared } from '../../src/lib/lossFunctions.mjs';

const trainingSet = [
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

export default class XORTrainer extends Trainer {
  constructor() {
    super({
      layers: {
        input: { numNeurons: 5 },
        hidden: [{ numNeurons: 2 }],
        output: { numNeurons: 1, skipBias: true }
      },
      examples: {
        trainSet: trainingSet,
        testSet: []
      },
      trainConfig: {
        lossFn: meanSquared,
        targetLoss: null
      }
    });
  }
}
