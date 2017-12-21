let neurons = 0;
let connections = 0;
let layerConnections = 0;
// types of connections
const LOGISTIC = (x, derivate) => {
    const fx = 1 / (1 + Math.exp(-x));
    if (!derivate)
      return fx;
    return fx * (1 - fx);
};

const activationCompute = (memo, {
			from: { activation }, weight, gain
		}) =>
			memo + activation * gain;

class JNeuron {
	static uid() {
		return neurons++;
	}
	static rand() {
		return Math.random() * .2 - .1;
	}
	constructor() {
		Object.assign(this, {
			ID: JNeuron.uid(),
			connections: {
				inputs: {},
				projected: {},
				gated: {}
			},
			activation: 0,
			state: 0,
			bias: JNeuron.rand()

		});
		this.project(this, 0);
	}

	activate() {
		const selfConnection = this.connections.projected[this.ID];
		// self math
		this.state = selfConnection.gain * selfConnection.weight * this.state * this.bias;
		const inputArray = Object.values(this.connections.inputs);
		// connections math
		//
		this.state = inputArray.reduce(activationCompute, this.state);

		this.activation = LOGISTIC(this.state);
		this.derivative = LOGISTIC(this.state, true);
	}

	project(to, weight) {
		const ID = connections++;
		const connection = {
			to,
			ID,
			from: this,
			weight: weight || JNeuron.rand(),
			gain: 1,
			gater: null
		};
		const {projected, inputs} = this.connections;
		projected[to.ID] = connection;
		inputs[to.ID] = connection;
		//this.connectionsMap[ID] = connection;
		// TRACE STUFF?
	}
}

class JInputNeuron extends JNeuron {

	activate(activation) {
		Object.assign(this, {
			activation,
			derivative: 0,
			bias: 0
		});
		return activation;
	}
}

// -------------------------------------
const connectionType = {
		ALL_TO_ALL: "ALL TO ALL",
		ONE_TO_ONE: "ONE TO ONE",
		ALL_TO_ELSE: "ALL TO ELSE"
	}

class JLayer {
	constructor(size = 0) {
		Object.assign(this, {
			list: Array(size).fill().map(() => new JNeuron()),
			connectedTo: []
		});
	}

	activate(input) {
		return this.list.map(neuron => neuron.activate());
	}

	project(to, assignedType, weights) {
		const { ONE_TO_ONE, ALL_TO_ALL } = connectionType;
		const type = assignedType || ALL_TO_ALL;

		this.connectedTo.push(to);

		if (type === ONE_TO_ONE) {
			if (from.length !== to.length) {
				throw new Error('One to one layer connection requre equal number of neurons per layer');
			}
			this.list.forEach((fromNeuron, index) => {
				const toNeuron = to.list[index];
				fromNeuron.project(toNeuron, weights);
			});
			return;
		}

		this.list.forEach(fromNeuron =>
			to.list.forEach(toNeuron =>
				fromNeuron.project(toNeuron, weights)
			)
		);
	}
}

class JInputLayer extends JLayer {

	constructor(size = 0) {
		super();
		this.list = Array(size).fill().map(() => new JInputNeuron());
	}

	activate(input) {
		if (input.length !== this.list.length)
			throw new Error('Input size and layer size must be equal to activate');
		return this.list.map((neuron, index) =>
			neuron.activate(input[index])
		);
	}
}

// --------------------------------------
class JNetwork {

	constructor({ input = null, hidden = [], output = null }) {
		Object.assign(this, {
			layers: { input, hidden, output }
		})
	}

	activate(inputs) {
		const { input, hidden, output } = this.layers;
		input.activate(inputs);
		[...hidden, output].forEach(layer => layer.activate());
	}

	propagate(rate, target) {
		const { output, hidden } = this.layers;
		output.propagate(rate, target);
		hidden.forEach(layer => layer.propagate(rate));
	}
}

//const jn = new JNeuron();
const input = new JInputLayer(2);
const hidden = new JLayer(3);
const output = new JLayer(1);
const jn = new JNetwork({ input, hidden: [ hidden ], output});
input.project(hidden);
hidden.project(output);

var learningRate = .3;
jn.activate([0,0]);
jn.propagate(learningRate, [0]);
output.list.forEach(n => console.log(n.connections.projected));
/*
// create the network
// var synaptic = require('synaptic');
// var Layer = synaptic.Layer;
// var Network = synaptic.Network;
import synaptic from 'synaptic';
const Layer = synaptic.Layer;
const Network = synaptic.Network;
var inputLayer = new Layer(2);
var hiddenLayer = new Layer(3);
var outputLayer = new Layer(1);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

var myNetwork = new Network({
	input: inputLayer,
	hidden: [hiddenLayer],
	output: outputLayer
});

// train the network - learn XOR
var learningRate = .3;
for (var i = 0; i < 200000; i++)
{
	// 0,0 => 0
	myNetwork.activate([0,0]);
	myNetwork.propagate(learningRate, [0]);

	// 0,1 => 1
	myNetwork.activate([0,1]);
	myNetwork.propagate(learningRate, [1]);

	// 1,0 => 1
	myNetwork.activate([1,0]);
	myNetwork.propagate(learningRate, [1]);

	// 1,1 => 0
	myNetwork.activate([1,1]);
	myNetwork.propagate(learningRate, [0]);
}

// test the network
console.log(myNetwork.activate([0,0])); // [0.015020775950893527]
console.log(myNetwork.activate([0,1])); // [0.9815816381088985]
console.log(myNetwork.activate([1,0])); // [0.9871822457132193]
console.log(myNetwork.activate([1,1])); // [0.012950087641929467]
*/
