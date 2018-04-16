import {JSparsemaxLayer} from './src/lib/layers.mjs';
const jsm = new JSparsemaxLayer()
console.log(jsm.activate([1.0, 2.0, 3.0, 9.9, 10.0]));
console.log(jsm.activate([3.0, -4.0, 2.0, 1.0, -9.0]));
