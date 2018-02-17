let numNeurons = 0;
let numConnections = 0;

class JNeuron {
  static uid() {
    numNeurons += 1;
    return numNeurons;
  }
  static rand() {
    return (Math.random() - 0.5);
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

  project(to) {
    numConnections += 1;
    const CID = numConnections;
    const { outs } = this;
    const connection = {
      to,
      CID,
      from: this,
      w: JNeuron.rand(),
      errorSum: 0
    };
    outs.push(connection);
    to.ins.push(connection);
    return connection;
  }

  activate() {
    this.s = this.ins.reduce((acc, { w, from: { z } }) =>
      acc + (z * w), 0);
    this.z = this.activationFunction(this.s);
    return this.z;
  }

  propagateSignal() {
    // todo, find a better way to do this
    // const { activationFunction } = this.outs[0].to;
    const weightAcc = this.outs.reduce((acc, { to, w }) => acc + (to.errorSignal * w), 0);
    this.errorSignal = this.activationFunction(this.z, true) * weightAcc;
    return this.errorSignal;
  }

  weighSumConnections() {
    const { outs, z } = this;
    outs.forEach((connection) => {
      connection.errorSum += (z * connection.to.errorSignal);
    });
  }

  toJSON() {
    return {
      NID: this.NID,
      z: this.z,
      s: this.s,
      errorSignal: this.errorSignal,
      bias: this.z === 1,
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
}

export class JBiasNeuron extends JNeuron {
  activate() {
    this.z = 1;
    return this.z;
  }
}

export class JInputNeuron extends JNeuron {
  activate(input) {
    this.z = input;
  }
}

export class JOutputNeuron extends JNeuron {
  propagateSignal(y) {
    const { z } = this;
    this.errorSignal = z - y;
    return this.errorSignal;
  }
}

export class JHiddenNeuron extends JNeuron {

}
