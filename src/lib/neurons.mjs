let numNeurons = 0;
let numConnections = 0;

class JNeuron {
  static uid() {
    numNeurons += 1;
    return numNeurons;
  }
  static rand() {
    return Math.random();
  }
  constructor({ activationFunction }) {
    Object.assign(this, {
      activationFunction,
      NID: JNeuron.uid(),
      outs: [],
      ins: [], // TODO WITH JSONDATA
      z: 0,
      s: 0
    });
  }

  project(to, weightMax) {
    numConnections += 1;
    const CID = numConnections;
    const { outs } = this;
    const connection = {
      to,
      CID,
      from: this,
      w: JNeuron.rand() * weightMax,
      errorSum: 0
    };
    outs.push(connection);
    to.ins.push(connection);
    return connection;
  }

  activate() {
    this.s = this.ins.reduce((acc, { CID, w, from: { NID, z } }) => {
      return acc + (z * w);
    }, 0.0);
    this.z = this.activationFunction.f(this.s);
    return this.z;
  }

  sumConnectionErrors() {
    return this.outs.reduce((acc, { to, w }) => {
      return acc + (to.errorSignal * w)
    } , 0.0);
  }

  calculateSignal() {
    this.errorSignal = this.activationFunction.d(this.z) * this.sumConnectionErrors();
  }

  updateConnectionSums() {
    const { z } = this;
    this.outs.forEach(connection => {
      connection.errorSum += z * connection.to.errorSignal;
    })
  }

  getConnectionSums(result) {
    return this.outs.forEach(connection => {
      result[connection.CID] = connection.errorSum;
    });
  }

  toJSON() {
    return {
      NID: this.NID,
      z: this.z,
      s: this.s,
      errorSignal: this.errorSignal,
      neuronType: this.constructor.name,
      outs: this.outs.map(({
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
      ins: this.ins.map(({
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
    this.s = this.z;
    this.z = 1;
    return this.z;
  }
  calculateSignal() {
    // Bias doesn't consider the derivitave of its 'value'
    this.errorSignal = this.sumConnectionErrors();
  }

}

export class JInputNeuron extends JNeuron {
  activate(input) {
    this.z = input;
    this.s = input;
  }
}

export class JOutputNeuron extends JNeuron {
  calculateLoss(y) {
    const { z } = this;
    // SQUARED ERROR
    // this.loss = 0.5 * Math.pow(y - z, 2);
    //this.loss = Math.abs(y- Math.round(z));
    //this.loss = z * Math.log(y) + (1-y) * Math.log(1-y);
    this.loss = y * Math.log(z) + (1-y) * Math.log(1-z);
    return this.loss;
  }
  calculateSignal(y) {
    const { z, NID } = this;
    // the reason we pass z to activationfunction derivative is because
    // it is less costly than calculating z again with the sums
    const errorSignal = ( y - z )
    //const errorSignal = (y - z) * this.activationFunction.d(z);
    this.errorSignal = errorSignal;
    return this.errorSignal;
  }


}

export class JHiddenNeuron extends JNeuron {

}
