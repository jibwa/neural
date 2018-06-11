import data from './diabetes.mjs';
import scalars from '../earlyproofstudy/resultScalarFunctions.mjs';

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

export default class DiabetesStudyParser {
  constructor() {
    const inputs = data.map(a => a.slice(0,8)) // features
    const scaledInputs = scalars.scaleFeatureData(inputs, 0, 1);
    const outputs = data.map(a => a.slice(8,9))
    this.trainingData = scaledInputs.map((row, index) => [row, outputs[index]]);


  }
  getData() {
    return this.trainingData;
  }
}
