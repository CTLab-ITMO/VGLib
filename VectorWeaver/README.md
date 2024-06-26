# VectorWeaver - безусловный генератор векторных изображений на основе двухуровневого трансформера.

## Установка

1) Установите зависимости:

```
pip install -r requirements.txt
```

Для установки cairosvg на Windows, запустите:

```
pip install cairosvg pipwin
pipwin install cairocffi
```

2) Скачайте [веса](https://drive.google.com/file/d/1Er9KdBcsSpmi8xDS1hYAOZYurMtjZiHo/view?usp=sharing).

## Генерация

Для генерации одного SVG изображения, запустите:

```
python -m tools.generate_svg --checkpoint [checkpoint_path] --output [output_image.svg]
```

Для генерации 9 SVG изображений за один подход, запустите:

```
python -m tools.generate_svg_3x3 --checkpoint [checkpoint_path] --output [output_image.png]
```

Где `checkpoint_path` - путь к скаченным весам

## Обучение

Для обучения модели требуется датасет с SVG изображениями.
Запустите модуль для конвертации изображений в формат, подходящий для модели:

```
python -m dataset.prepare_dataset --data_folder [folder with .svg files] --output_folder [folder for prepared dataset]
```

Для добавления аугментации к датасету, запустите следующую команду:

```
python -m dataset.extend_with_augmentations --data_folder [output folder from last step] --output_folder [folder for prepared dataset with augmentations]
```

Количество и список аугментаций может быть изменен в исходном коде.

Вы можете проверить корректность конвертации с помощью команды:

```
python -m tools.draw_from_dataset --input [our format file] --output [resulting .svg file]
```

Для запуска обучения автоэнкодера запустите:

```
python -m train.train_vae --input [prepared dataset] --checkpoint_output [resulting models]
```

Вы можете продолжить обучение с контрольной точки, указав опцию `--checkpoint_input`.

Вы можете изменить параметры модели, переопределив их в `configs/config.py`.

Для проверки качества автоэнкодера используйте модуль, который применяет модель к набору данных:

```
python -m tools.apply_autoencoder --input [our format file] --checkpoint [checkpoint path] --output [resulting .svg file]
```

Для запуска обучения диффузионной модели, запустите:

```
python -m train.train_diffusion --input [prepared dataset] --checkpoint_input [checkpoint with VAE] --checkpoint_output [resulting models]
```

Теперь вы можете генерировать изображения на основе приведенной выше инструкции!

