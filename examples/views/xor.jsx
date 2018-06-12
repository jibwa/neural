import React from 'react';
import PropTypes from 'prop-types';

const OutVis = ({ z, w }, iterator) => (
  <td key={`out${iterator}`}>
    w:{w}
  </td>
);

OutVis.propTypes = {
  z: PropTypes.number.isRequired,
  w: PropTypes.number.isRequired
};

const Neuron = ({ neuron }) => {
  const { z, s } = neuron;
  const { outs } = neuron.conns();
  return (
    <table>
      <tr>
        <td>
          i:{z}   s:{s}
        </td>
      </tr>
      <tr>
        {outs.map(OutVis)}
      </tr>
    </table>
  );
};
Neuron.propTypes = {
  z: PropTypes.number.isRequired,
  s: PropTypes.number.isRequired
};

const Layer = ({ neurons }) => (
  <div>
    {neurons.map((neuron, iterator) => (
      <Neuron key={`n${iterator}`} neuron={neuron} />
    ))}
  </div>
);
Layer.propTypes = {
  neurons: PropTypes.arrayOf(PropTypes.shape).isRequired
};

const Visualize = ({
  network: {
    layers: { input, hidden, output }
  },
  trainingSet,
  preds,
  totalIterations
}) => (
  <html lang="en">
    <head>
      <title>Network Visualizer</title>
      <link rel="stylesheet" href="/style/style.css" />
    </head>
    <body>
      <h2>Network Output</h2>
      <h4>Total Iterations: {totalIterations}</h4>
      <h3>Preds</h3>
      <table>
        <tr>
          <th>
            Input
          </th>
          <th>
            Expected Output
          </th>
          <th>
            Neural Prediction
          </th>
        </tr>
        {preds.map(([feature, label, y]) => (
          <tr>
            <td>
              {feature}
            </td>
            <td>
              {label}
            </td>
            <td>
              {y}
            </td>
          </tr>
        ))}
      </table>
      <h3>Input</h3>
      <Layer neurons={input.neurons} />
      <h3>Hidden</h3>
      {hidden.map(({ neurons }, iterator) => (
        <Layer key={`l${iterator}`} neurons={neurons} />
      ))}
      <h3>Output</h3>
      <Layer neurons={output.neurons} />
    </body>
  </html>
);

Visualize.propTypes = {
  network: PropTypes.shape.isRequired
};

export default Visualize;
