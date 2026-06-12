# Backpropagation of Love ❤️

A small Valentine's Day experiment where a simple neural network learns, step by step, to draw a heart.

Built with Python, NumPy and Matplotlib, with backpropagation implemented manually.

## Output

The project generates:

- a static PNG image
- an animated GIF showing the learning process

## Example

The final GIF is available here:

```text
amor_animado.gif
```

The final static image is available here:

```text
amor_tracados.png
```

## Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
python amor.py
```

After running the script, the generated files will be saved inside:

```text
saida_amor/
```

## How it works

The script creates a heart-shaped target curve and trains a small neural network to approximate it.

The network uses:

- sigmoid activations
- mean squared error as the cost function
- manual backpropagation
- gradient descent
- NumPy for matrix operations
- Matplotlib for visualization
- Pillow for GIF generation

No deep learning framework is used.

## Notes

This is a small educational and visual experiment.

The goal is not to build a production model, but to show the learning process visually: the curve starts almost random, the error decreases, the weights adjust, and the heart gradually appears.


## Inspiration

This project was inspired by a learning exercise I completed during my studies at King's College London.

The implementation, visual design and Valentine's Day adaptation were developed independently as a personal experiment.
