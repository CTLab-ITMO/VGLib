# EvoVec: Evolutionary Image Vectorization with Adaptive Curve Number and Color Gradients

Контакты: [Егор Баженов](tujh.bazhenov.kbn00@mail.ru)

## Примеры работы алгоритма

| Исходное изображение     | ![](data/test%20images/readme%20examples/init_hippo.png)    | ![](data/test%20images/readme%20examples/init_land.png)    | ![](data/test%20images/readme%20examples/init_list.png)    | ![](data/test%20images/readme%20examples/init_monkey.png)    | ![](data/test%20images/readme%20examples/init_smile.png)    |
|------------------|-------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------|-------------------------------------------------------------|
| Векторизованное изображение | ![](data/test%20images/readme%20examples/my_algo_hippo.png) | ![](data/test%20images/readme%20examples/my_algo_land.png) | ![](data/test%20images/readme%20examples/my_algo_list.png) | ![](data/test%20images/readme%20examples/my_algo_monkey.png) | ![](data/test%20images/readme%20examples/my_algo_smile.png) |

Мы представляем новый метод векторизации изображений с использованием переменного числа траекторий, основанный на эволюционном алгоритме.
Результат детерминированного алгоритма выбирается в качестве начальной совокупности. Далее для получения лучшего векторизованного изображения итеративно применяются различные мутации и кроссинговеры.

## Использование

1. ``git clone https://github.com/EgorBa/EvoVec-Evolutionary-Image-Vectorization``
2. ``pip install requirements.txt``
3. Конфигурация [config file](config.py) для вашей задачи

#### Config parameters description

| Название параметра   | Описание                                                         | Тип                      |
|------------------|---------------------------------------------------------------------|---------------------------------|
| DEBUG            | Требуется показать отладочную информацию                                                | Boolean                         |
| PNG_PATH         | Путь к png-файлу для векторизации                                 | String                          |
| INDIVIDUAL_COUNT | Количество особей в популяции                                 | Int                             |
| ELITE_PERCENT    | Процент популяции, который должен остаться для следующей итерации                     | Float                           |
| STEP_EVOL        | Количество эпох                                      | Int                             |
| FITNESS_TYPE     | Тип функции отбора                                          | [Fitness](fitness/loss_type.py) |
| MUTATION_TYPE    | Массив с используемыми мутациями                                             | Array<[Mutation](mutations)>    |
| CROSSOVER        | Массив с используемыми кроссоверами                                            | Array<[Crossover](crossover)>   |
| COLOR_DIFF       | Максимальная разница в цвете между пикселями, которая должна быть скорректирована | Int                             |
| MAX_W, MAX_H     | Максимальное значение ширины и высоты векторизованного изображения       | Int                             |

4. ``python main.py``
