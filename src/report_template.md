## Цель и задачи
Необходимо по имеющимся данным о ценах на жильё предсказать окончательную цену каждого дома с учетом характеристик домов с использованием нейронной сети.
Задачи:
* создать и инициализировать последовательную модель нейронной сети с помощью фрэймворков тренировки нейронных сетей
* перебрать различные параметры нейросети, выбрать наиболее качественный набор
* с помощью нейросети с выбранным набором параметров оценить тестовые данные
* сделать выводы касательно влияния различных параметров на качество работы нейросети

## Исходные данные
Описание набора данных содержит 80 классов (набор переменых) классификации оценки типа жилья. Даны тренировочные данные, для которых известен параметр цены, и даны тестовые данные, для которых необходимо определить параметр цены.

## Процесс выполнения
В качестве фреймворка был выбран **tensorflow**. В его основе лежит компилируемый (статический) граф вычислений, поэтому ожидалась большая скорость выполнения задачи по сравнению с **torch**, который использует динамический граф вычислений.

Для удобства тестирования алгоритм решения был разделен на несколько частей - **скриптов python**. Их объединяет **Makefile**. Теперь при модификации одного из скриптов будет необходимо заново исполнить только его и зависимые скрипты.

## Полученные результаты

GRAPHICS_HERE

### Оценка тестовых данных
Прилагается в файле **best_answers.csv**.

## Выводы
