import express from 'express';
import expressLess from 'express-less';
import erv from 'express-react-views';

import XORTrainer from './src/XORTrainer.mjs';
import TrainingMonitor from './src/TrainingMonitor.mjs';
// let data = JSON.stringify(input, true);
// fs.writeFileSync('input.json', data);

const trainer = new XORTrainer();
const { network, trainingSet } = trainer;

// trainer.train(5000);
const app = express();
if (process.version !== 'v9.3.0') {
  console.log(`This application is written for node v9.3.0, you are running: ${process.version}`);
  console.log('Please install the correct version with NVM');
  console.log('Attempting to start the APP. USE AT YOUR OWN RISK');
}

app.set('views', './views');
app.set('view engine', 'jsx');
app.engine('jsx', erv.createEngine());

app.use('/style', expressLess('./less', { debug: true }));

const renderIndex = (req, res) => {
  res.render('index', { network, trainingSet });
};

app.get('/', (req, res) => res.json(network.toJSON()));
app.get('/visualize', renderIndex);
app.get('/visualize/:iterations', (req, res) => {
  const iterations = parseInt(req.params.iterations, 10);
  trainer.train(iterations);
  renderIndex(req, res);
});
let monitor;
const renderMonitor = (req, res, iterations) => {
  if (!monitor) {
    monitor = new TrainingMonitor(trainer);
  }
  res.json(monitor.monitor(iterations));
};
app.get('/monitor', (req, res) => renderMonitor(req, res, 0));
app.get('/monitor/:iterations', (req, res) => {
  const iterations = parseInt(req.params.iterations, 10);
  renderMonitor(req, res, iterations);
});

app.listen(4000, () => {
  console.log('Example app listening on port 4000!');
  console.log('JSON VIEW: http://localhost:4000');
  console.log('TRAIN/VISUALIZE (1000 iterations): http://localhost:4000/visualize/1000');
});
