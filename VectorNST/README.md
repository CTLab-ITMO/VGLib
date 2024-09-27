# VectorNST

<div align="center">
  <img src="images/owl_stylized_owl1.jpg" alt="img1" width="512" height="160"/>
  <img src="images/scene6_stylized_scene8.jpg" alt="img2" width="512" height="130"/>
  <img src="images/flower.jpg" alt="img3" width="512" height="160"/>
</div>

Официальная реализация модели VectorNST (векторный перенос стиля) на PyTorch.

[Демо сервис](http://81.3.154.178:5001/vector_style_transfer).

## Использование

### Требования

Установите следующие зависимости:

* [PyTorch](https://pytorch.org) и Torchvision
* Pillow
* NumPy
* [DiffVG](https://github.com/IzhanVarsky/diffvg) и его зависимости

Пример установки:

* `pip install torch torchvision pillow numpy`
* `pip install cssutils scikit-learn scikit-image svgwrite svgpathtools matplotlib`
* `git clone --recursive https://github.com/IzhanVarsky/diffvg`
* `cd diffvg && python ./setup.py install && cd ..`

### Тестирование

Запустите наш пример `python test_vector_nst.py --content_img ./images/owl.svg --style_img ./images/owl1.jpg`.

Вы также можете указать другие параметры: запустите `python test_vector_nst.py --help`, чтобы увидеть больше информации.

### JointLoss

Механизм работы [JointLoss](joint_loss.py) основан на принципе поиска точек стыка у кубических кривых Безье, являющихся
одним из основных компонентов SVG-формата изображения. Точки стыка - точки, в которых кривая
не имеет касательной к своей поверхности. В таком
случае чем больше кривизна кривой в малой окрестности данной точки,
тем более неровным получается контур выходного изображения.

Основная идея поиска точек стыка основана на анализе производной вектор-функции, описывающей кривую:
в подобных точках производная не определена, а также в них находится локальный
экстремум функции.

Вы можете ознакомиться с более подробной информацией в [этом файле](JointLoss.pdf)