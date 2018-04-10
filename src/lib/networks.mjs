import { JInputLayer, JOutputLayer, JHiddenLayer } from './layers.mjs';

export class JNetwork {
  constructor(layerDef) {
    const input = new JInputLayer(layerDef.input);
    const hidden = layerDef.hidden.map(layer => new JHiddenLayer(layer));
    const output = new JOutputLayer(layerDef.output);

    [...hidden, output].reduce((curr, next) => {
      curr.project(next);
      return next;
    }, input);

    Object.assign(this, {
      layers: { input, hidden, output }
    });
  }

  dropout() {
    const { input, hidden } = this.layers;
    [input, ...hidden].forEach(layer => {
      const { dropout } = layer.layerDef;
      if (!dropout) {
        return;
      }
      layer.neurons.forEach(neuron => {
        if (Math.random() > dropout) {
          neuron.drop()
        }
      }, [])
    });
  }
  restoreDropout() {
    const { input, hidden } = this.layers;
    [input, ...hidden].forEach(layer => {
      const { dropout } = layer.layerDef;
      if (!dropout) {
        return;
      }
      layer.neurons.forEach(neuron => {
        neuron.restoreDrop();
      })
    });
  }

  activate(inputs, testing) {
    const { input, hidden, output } = this.layers;
    input.activate(inputs, true);
    hidden.map(layer => layer.activate());
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
    this.getAllNeurons().forEach(neuron =>
      neuron.outs.forEach(connection => {
        connection.errorSum = 0;
      }));
  }

  getConnectionSums() {
    const result = {};
    this.getFlatLayers().forEach(layer => layer.getConnectionSums(result));
    return result;;
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
    this.getFlatLayers().forEach(layer =>
      allNeurons.push(...layer.neurons));
    return allNeurons;
  }

  getAllConnections() {
    const connections = [];
    this.getAllNeurons().forEach(neuron =>
      connections.push(...neuron.ins));
    return connections;
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
    return this.getFlatLayers().map(layer => layer.getWeightArrays());
  }
  restoreWeights(weights) {
    this.getFlatLayers.forEach((layer, i) => layer.restoreWeights(weights[i]));
  }
}
