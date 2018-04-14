import {
  JInputNeuron,
  JHiddenNeuron,
  JOutputNeuron,
  JBiasNeuron,
  JSoftmaxXENeuron
} from './neurons.mjs';
import { toJSON } from '../helpers.mjs';
import { sigmoid, softmax } from './activationFunctions.mjs';

class JLayer {
  constructor(layerDef, NeuronClass, cm, layerInt) {
    const {
      numNeurons,
      activationFunction = sigmoid,
      weightMax = 1.0,
      skipBias = false,
      dropout = 1.0,
      wBound,
    } = layerDef;
    this.connectedTo = [];

    let bias;
    if (!skipBias) {
      bias = new JBiasNeuron({ activationFunction, wBound , cm });
    }
    const neurons = Array(numNeurons).fill().map(() => new NeuronClass({ activationFunction, cm, wBound }));
    Object.assign(this, {
      neurons: bias ? [bias, ...neurons] : neurons,
      connectedTo: [],
      dropouts: [],
      layerDef,
      layerInt,
      cm
    });
  }

  activate(testing) {
    return this.neurons.map(neuron => neuron.activate(testing));
  }

  // set up the neuron relationships
  project(to) {
    const {layerInt, layerDef: { weightMax } } = this;
    this.connectedTo.push(to);

    this.neurons.forEach((fromNeuron) => {
      to.neurons.forEach((toNeuron) => {
        if (!(toNeuron instanceof JBiasNeuron)) {
          fromNeuron.project(toNeuron, weightMax, layerInt);
        }
      });
    });
  }
  calculateSignal() {
    return this.neurons.map(neuron => neuron.calculateSignal());
  }
  updateConnectionSums() {
    return this.neurons.map(neuron => neuron.updateConnectionSums())
  }
  toJSON() {
    return {
      layerType: this.constructor.name,
      neurons: this.neurons.map(toJSON)
    };
  }
  getWeightArrays() {
    return this.neurons.map(neuron => neuron.getWeightArray());
  }
  restoreWeights(weights) {
    return this.neurons.forEach((neuron, i) => neuron.restoreWeights(weights[i]));
  }

  drop(modifyNeurons) {
    const { layerDef: { dropout }, dropouts, neurons } = this;
    if (!dropout) {
      return;
    }
    const newNeurons = this.neurons.reduce((acc, neuron) => {
      if (Math.random() > dropout) {
        // bias nodes wont let themselves be dropped and report false
        const dropped = neuron.drop();
        if (dropped) {
          dropouts.push(neuron);
          if (modifyNeurons) {
            return acc;
          }
        }
      }
      acc.push(neuron);
      return acc;
    }, []);
    this.neurons = newNeurons;
  }
  restoreDrop(modifyNeurons) {
    const { neurons, layerDef: { dropout }, dropouts } = this;
    if (!dropout) {
      return;
    }
    this.dropouts.forEach(neuron => {
      neuron.restoreDrop();
    })
    if (modifyNeurons) {
      neurons.push(...dropouts);
    }
    this.dropouts = [];
  }
}
export class JOutputLayer extends JLayer {
  constructor(layerDef, cm, layerInt, NeuronType=JOutputNeuron) {
    super(layerDef, NeuronType, cm);
  }

  calculateSignal(target) {
    if (this.neurons.length !== target.length) {
      throw new Error(`TARGET size and LAYER size must be the same to propagate. Usually this means there is an unexpected bios node. Target size: ${target.length}, Layer size: ${this.neurons.length}`);
    }
    return this.neurons.map((neuron, index) => neuron.calculateSignal(target[index]));
  }
}
export class JSoftmaxXEOutputLayer extends JOutputLayer {
  constructor(layerDef, cm, layerInt) {
    super({
      ...layerDef,
      activationFunction: softmax,
      skipBias: true
    },
      cm,
      layerInt,
      JSoftmaxXENeuron,
    );
  }
  activate() {
    const ts = this.neurons.map(n => n.activate());
    const sum = ts.reduce((acc, t) => acc + t);
    const zs = this.neurons.map(n => {
      n.z = n.z / sum
      return n.z;
    });
    return zs;
  }

}

export class JHiddenLayer extends JLayer {
  constructor(layerDef, cm, layerInt) {
    super(layerDef, JHiddenNeuron, cm, layerInt);
  }
  drop() {
    super.drop(true);
  }

  restoreDrop() {
    super.restoreDrop(true)
  }
}

export class JInputLayer extends JLayer {
  constructor(layerDef, cm, layerInt) {
    super(layerDef, JInputNeuron, cm, layerInt);
  }

  activate(inputs, testing) {
    const { neurons, layerDef: { dropout } } = this;
    const inputNeurons = neurons.filter(neuron => neuron instanceof JInputNeuron);
    const biasNeurons = neurons.filter(neuron => neuron instanceof JBiasNeuron);
    if (inputs.length !== inputNeurons.length) {
      throw new Error(`Input size and layer size must be equal to activate  I:${inputs.length}  N:${this.neurons.length}`);
    }
    let dropoutP = 1.0;
    if (testing === true) {
      dropoutP = dropout;
    }
    biasNeurons.forEach(neuron => neuron.activate());
    inputNeurons.forEach((neuron, index) => {
      neuron.activate(inputs[index] * dropoutP)
    });
  }
}
