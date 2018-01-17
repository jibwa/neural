import React from 'react';
import PropTypes from 'prop-types';


const OutVis = ({ input, weight }, iterator) => (
  <td key={`out${iterator}`}>
    {input}/w:{weight}
  </td>
);
OutVis.propTypes = {
  input: PropTypes.number.isRequired,
  weight: PropTypes.number.isRequired
};

const Neuron = ({ input, outs, sum }) => (
  <table>
    <tr>
      <td>
        {input}/s:{sum}
      </td>
    </tr>
    <tr>
      {outs.map(OutVis)}
    </tr>

  </table>
);
Neuron.propTypes = {
  input: PropTypes.number.isRequired,
  outs: PropTypes.arrayOf(PropTypes.shape).isRequired,
  sum: PropTypes.number.isRequired
};

const Layer = ({ neurons }) => (
  <div>
    {neurons.map((neuron, iterator) => (
      <Neuron key={`n${iterator}`} {...neuron} />
    ))}
  </div>
);
Layer.propTypes = {
  neurons: PropTypes.arrayOf(PropTypes.shape).isRequired
};

const Visualize = ({
  network: {
    layers: { input, hidden, output }
  }
}) => (
  <html lang="en">
    <head>
      <title>Network Visualizer</title>
      <link rel="stylesheet" href="/style/style.css" />
    </head>
    <body>
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
