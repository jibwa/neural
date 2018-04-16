let numNeurons = 0;
let numConnections = 0;

const sumZW = (acc, { w, from: { z } }) => {
  return acc + (z * w);
}
const bakSumZW = (acc, { w, from: { z } }) => {
  return acc + (z * w);
}

class JNeuron {
  static uid() {
    numNeurons += 1;
    return numNeurons;
  }
  static rand() {
    return Math.random();
  }
  constructor({ activationFunction, wBound, cm }) {
    Object.assign(this, {
      activationFunction,
      wBound,
      NID: JNeuron.uid(),
      cm,
      z: 0,
      s: 0
    });
  }
  conns() {
    return this.cm.getNodeConnections(this.NID);
  }

  project(to, weightMax, layerInt) {
    const from = this;
    const w = JNeuron.rand() * weightMax;
    const errorSum = 0;
    this.cm.register({ to, from, w, errorSum, layerInt })
  }

  activate() {
    const { ins } = this.conns();
    const s = ins.reduce(sumZW, 0.0);
    this.z = this.activationFunction.f(s);
    return this.z;
  }

  sumConnectionErrors() {
    const { outs } = this.conns();
    return outs.reduce((acc, { to, w }) => {
      return acc + (to.errorSignal * w)
    } , 0.0);
  }

  calculateSignal() {
    this.errorSignal = this.activationFunction.d(this.z) * this.sumConnectionErrors();
  }

  updateConnectionSums() {
    const { z } = this;
    const { outs } = this.conns();
    outs.forEach(connection => {
      connection.errorSum += z * connection.to.errorSignal;
    })
  }

  wBoundWeight(connection) {
    const { wBound } = this;
    if (wBound) {
      if (connection.w > wBound) {
        connection.w = wBound;
      }
      if (connection.w < -wBound) {
        connection.w = -wBound;
      }
    }
  }
  updateWeights(learningRate, batchSize, { level, lambda }) {
    const { outs } = this.conns();
    const pBSize = parseFloat(batchSize);
    outs.forEach(connection => {
      const { errorSum, w } = connection;
      const d = -learningRate * (1.0 / pBSize) * errorSum
      if (!(lambda > 0.0)) {
        // no regularization, simple update
         connection.w -= d
      } else {
        if (level === 1) {
          const sgn = w < 0 ? -1.0 : 1.0;
          const reg = ((learningRate * lambda ) / pBSize) * sgn;
          connection.w = w - reg - d;

        } else if (level === 2) {
          const reg = 1.0 - (( learningRate * lambda ) / pBSize);
          connection.w =  reg * w - d;
        }
      }
      this.wBoundWeight(connection);
      connection.errorSum = 0;
    });
  }

  drop() {
    return this.cm.dropoutNeuron(this.NID);
  }

  restoreDrop() {
    this.cm.restoreNeuron(this.NID);
  }

  toJSON() {
    const { outs, ins } = this.conns();
    return {
      NID: this.NID,
      z: this.z,
      errorSignal: this.errorSignal,
      neuronType: this.constructor.name,
      outs: outs.map(({
        CID,
        to: { NID },
        w,
        errorSum
      }) => ({
        NID,
        w,
        CID,
        errorSum
      })),
      ins: ins.map(({
        CID,
        from: { NID },
        w,
        errorSum
      }) => ({
        NID,
        w,
        CID,
        errorSum
      }))
    };
  }
  getWeightArray() {
    return this.outs.map(out => out.w);
  }
  restoreWeights(weights) {
    this.outs.map((out, i) => out.w = weights[i]);
  }
}

export class JBiasNeuron extends JNeuron {
  activate() {
    this.z = 1;
    return this.z;
  }
  calculateSignal() {
    // Bias doesn't consider the derivitave of its 'value'
    this.errorSignal = this.sumConnectionErrors();
  }
  updateWeights(learningRate, batchSize) {
    this.conns().outs.forEach(connection => {
      connection.w += learningRate * ((1 / batchSize) * connection.errorSum);
      this.wBoundWeight(connection);
      connection.errorSum = 0;
    });
  }
  // DONT dropout bias
  drop() {
    return false;
  }

  restoreDrop() {
    throw 'Bias should not have been dropped';
  }
}

export class JInputNeuron extends JNeuron {
  activate(input) {
    this.z = input;
  }
}

export class JOutputNeuron extends JNeuron {
  calculateSignal(y) {
    // no dropout here necause you cant drop output neurons
    const { z, NID } = this;
    // the reason we pass z to activationfunction derivative is because
    // it is less costly than calculating z again with the sums
    const errorSignal = ( y - z )
    //const errorSignal = (y - z) * this.activationFunction.d(z);
    this.errorSignal = errorSignal;
    return this.errorSignal;
  }
}

export class JSoftmaxXENeuron extends JOutputNeuron {
}

export class JHiddenNeuron extends JNeuron {

}
