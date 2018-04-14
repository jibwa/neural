import { JInputLayer, JOutputLayer, JSoftmaxXEOutputLayer,  JHiddenLayer } from './layers.mjs';
import { ConnectionManager } from './connectionManager.mjs';

export class JNetwork {
  constructor(layerDef) {
    const cm = new ConnectionManager();
    const input = new JInputLayer(layerDef.input, cm, 0);
    let layerInt = 1;
    const hidden = layerDef.hidden.map(layer => {
      let hl = new JHiddenLayer(layer, cm, layerInt)
      layerInt += 1
      return hl;
    });
    const OutputLayerClass = layerDef.output.softmaxXE === true ?
      JSoftmaxXEOutputLayer : JOutputLayer;
    const output = new OutputLayerClass(layerDef.output, cm, layerInt);

    [...hidden, output].reduce((curr, next) => {
      curr.project(next);
      return next;
    }, input);

    Object.assign(this, {
      cm,
      layers: { input, hidden, output }
    });
  }

  dropout() {
    const { input, hidden } = this.layers;
    [input, ...hidden].forEach(layer => layer.drop())
  }
  restoreDropout() {
    const { input, hidden } = this.layers;
    [input, ...hidden].forEach(layer => layer.restoreDrop())
  }
  scaleDropoutPForPrediction(invert) {
    const { cm, layers: { input, hidden } } = this;
    [input, ...hidden].forEach(({layerInt, layerDef: { dropout } }) =>
      cm.scaleWeightsForDropout(layerInt, dropout, invert) );

  }

  activate(inputs, testing) {
    const { input, hidden, output } = this.layers;
    input.activate(inputs, testing);
    hidden.map(layer => layer.activate(testing));
    return output.activate();
  }

  calculateSignal(ys) {
    const { layers: { hidden, input, output } } = this;
    output.calculateSignal(ys);
    return [...hidden.slice().reverse(), input].forEach(layer => layer.calculateSignal());
  }

  updateConnectionSums() {
    // TODO REMOVE THIS FUNCTION NOT USED
    const { layers: { input, hidden } } = this;
    return [input, ...hidden].map(layer => layer.updateConnectionSums());
  }
  // We are going to start a new training set or group
  clearConnectionSums() {
    // eslint-disable-next-line no-return-assign
    this.cm.allConnections().forEach(connection => connection.errorSum = 0);
  }
  getConnections() {
    return this.cm.allConnections(true);
  }
  getConnectionSums() {
    return this.cm.allConnections(true).map(({ CID, errorSum }) => [CID, errorSum]);
  }

  updateWeights(learningRate, batchSize, regularize) {
    // TODO - BIASES USE A DIFFERENT FORMULA
    this.getAllNeurons().forEach(neuron => neuron.updateWeights(learningRate, batchSize, regularize));
    //this.getAllConnections().forEach((connection) => {
    //  connection.w += ((1 / batchSize) * connection.errorSum);
    //  connection.errorSum = 0;
    //});
  }

  getAllNeurons() {
    const allNeurons = [];
    this.getFlatLayers().forEach(layer => {
      allNeurons.push(...layer.neurons)
    });
    return allNeurons;
  }

  getAllConnections() {
    return this.cm.allConnections();
  }

  getFlatLayers() {
    const { input, hidden, output } = this.layers;
    return [input, ...hidden, output];
  }

  // inputs is optinally training set inputs to add to json
  toJSON(trainingSet = []) {
    return {
      inputOutput: trainingSet.map(([input, output]) => [input, this.activate(input), output]),
      layers: this.getFlatLayers().map(layer => layer.toJSON())
    };
  }
  getWeightArrays() {
    return this.cm.layeredConnectionWeights();
  }
  restoreWeights(weights) {
    this.getFlatLayers.forEach((layer, i) => layer.restoreWeights(weights[i]));
  }
}
