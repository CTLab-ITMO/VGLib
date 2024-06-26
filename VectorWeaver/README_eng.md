# VectorWeaver - an unconditional generator of vector images based on a two-stage transformer.

## Installation

1) Install the dependencies:

```
pip install -r requirements.txt
```

To install cairosvg on Windows, you need to run:

```
pip install cairosvg pipwin
pipwin install cairocffi
```

2) Download [the checkpoint](https://drive.google.com/file/d/1Er9KdBcsSpmi8xDS1hYAOZYurMtjZiHo/view?usp=sharing).

## Generation

To generate one SVG image, run the module:

```
python -m tools.generate_svg --checkpoint [checkpoint_path] --output [output_image.svg]
```

[checkpoint_path] - path to the downloaded checkpoint

To generate 9 SVG images at once, you can use the module:

```
python -m tools.generate_svg_3x3 --checkpoint [checkpoint_path] --output [output_image.png]
```

## Train your own model

To train your model, you will need a dataset with SVG images. Run this module to convert images into a format
appropriate for the model:

```
python -m dataset.prepare_dataset --data_folder [folder with .svg files] --output_folder [folder for prepared dataset]
```

To add augmentation to the dataset, you need to run the following module:

```
python -m dataset.extend_with_augmentations --data_folder [output folder from last step] --output_folder [folder for prepared dataset with augmentations]
```

The number and list of augmentations can be changed in the source code.

You can check the correctness of the conversion using the module:

```
python -m tools.draw_from_dataset --input [our format file] --output [resulting .svg file]
```

To start the autoencoder training run:

```
python -m train.train_vae --input [prepared dataset] --checkpoint_output [resulting models]
```

You can continue training from the checkpoint by specifying "--checkpoint_input" option.
You can change the model parameters by overriding them in "configs/config.py".

To check the quality of the autoencoder, use the module that applies the model to the dataset:

```
python -m tools.apply_autoencoder --input [our format file] --checkpoint [checkpoint path] --output [resulting .svg file]
```

To start the diffusion training run:

```
python -m train.train_diffusion --input [prepared dataset] --checkpoint_input [checkpoint with VAE] --checkpoint_output [resulting models]
```

Now you can generate images according to the instructions above!

