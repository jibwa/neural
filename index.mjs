import express from 'express';
import expressLess from 'express-less';
import erv from 'express-react-views';

import XORTrainer from './src/XORTrainer.mjs';
// let data = JSON.stringify(input, true);
// fs.writeFileSync('input.json', data);

const trainer = new XORTrainer();
const { network, trainingSet } = trainer;

// trainer.train(5000);
const app = express();
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

app.listen(4000, () => {
  console.log('Example app listening on port 4000!');
  console.log('JSON VIEW: http://localhost:4000');
  console.log('TRAIN/VISUALIZE (1000 iterations): http://localhost:4000/visualize/1000');
});
