# Backpropagation of Love ❤️

A small Valentine's Day experiment where a simple neural network learns, step by step, to draw a heart.

Built with Python, NumPy and Matplotlib, with backpropagation implemented manually.

## Output

The project generates:

- a static PNG image
- an animated GIF showing the learning process

## Example

The final GIF is generated at:

```text
saida_amor/amor_animado.gif
```

The final static image is generated at:

```text
saida_amor/amor_tracados.png
```

## Run

Install the dependencies:

```bash
pip install numpy matplotlib pillow
```

Run the script:

```bash
python amor.py
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
