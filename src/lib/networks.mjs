import { JInputLayer, JOutputLayer, JHiddenLayer } from './layers.mjs';

export class JNetwork {
  constructor(layerDef, jsonData) {
    let input;
    let hidden;
    let output;
    const { learningRate } = layerDef;

    if (jsonData) {
      input = new JInputLayer(null, jsonData.input);
      hidden = jsonData.hidden.map(layer => new JHiddenLayer(null, layer));
      output = new JOutputLayer(null, jsonData.output);
    } else {
      input = new JInputLayer(layerDef.input);
      hidden = layerDef.hidden.map(num => new JHiddenLayer(num));
      output = new JOutputLayer(layerDef.output);
      const all = [...hidden, output];
      all.reduce((curr, next) => {
        curr.project(next);
        return next;
      }, input);
    }
    Object.assign(this, {
      layers: { input, hidden, output },
      learningRate
    });
  }

  activate(inputs) {
    const { input, hidden, output } = this.layers;
    input.activate(inputs);
    [...hidden].forEach(layer => layer.activate());
    return output.activate();
  }

  propagate(target) {
    const { layers: { output, hidden }, learningRate } = this;
    output.propagate(target, learningRate);
    hidden.forEach(layer => layer.propagate(learningRate));
  }
  toJSON() {
    const { input, hidden, output } = this.layers;
    return {
      layers: [input, ...hidden, output].map(layer => layer.toJSON())
    };
  }
}
