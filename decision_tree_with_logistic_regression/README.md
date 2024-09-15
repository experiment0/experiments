# Дерево решений с использованием логистической регрессии в качестве разделяющей плоскости

## Оглавление

[Описание проекта](https://github.com/experiment0/experiments/tree/master/decision_tree_with_logistic_regression#Описание-проекта)\
[Краткая информация о данных](https://github.com/experiment0/experiments/tree/master/decision_tree_with_logistic_regression#Краткая-информация-о-данных)\
[Этапы работы над проектом](https://github.com/experiment0/experiments/tree/master/decision_tree_with_logistic_regression#Этапы-работы-над-проектом)\
[Описание файлов](https://github.com/experiment0/experiments/tree/master/decision_tree_with_logistic_regression#Описание-файлов)\
[Результат](https://github.com/experiment0/experiments/tree/master/decision_tree_with_logistic_regression#Результат)

## Описание проекта

Для закрепления понимания работы алгоритма дерева решений он реализован вручную.\
Отдельные функции для его реализации были даны в курсе [Профессия Data Scientist](https://skillfactory.ru/data-scientist-pro).\
Возникло желание собрать их в класс, прописать типы и добавить комментарии.\
В процессе реализации возникла идея взять в качестве разделяющей плоскости не предикат, а логистическую регрессию.\
Было предположение, что прирост информации после разделения с помощью логистической регрессии будет больше,
чем с помощью разделения по предикату.\
А это в свою очередь даст лучший результат на каждом шаге и как следствие на выходе.

## Краткая информация о данных

## Этапы работы над проектом

- В файле `DecisionTree.py` вручную реализован алгоритм дерева решений для регрессии и классификации.
- Проведено сравнение результатов его работы с классами `DecisionTreeRegressor` и `DecisionTreeClassifier` из `sklearn.tree`.
- В файле `DecisionTreeWithLogisticRegression.py` аналогичным образом реализован алгоритм дерева решений,
  но в качестве плоскости, разделяющей данные на 2 выборки на каждом шаге, взят не предикат, а логистическая регрессия.
- Проведено сравнение результатов созданного алгоритма и `DecisionTreeRegressor` из `sklearn.tree`.\
  А также проведено сравнение взвешенной неоднородности после разделения выборок на первом шаге для обоих алгоритмов.
- Сделан вывод.

## Описание файлов

## Результат

:arrow_up:[к оглавлению](https://github.com/experiment0/experiments/tree/master/decision_tree_with_logistic_regression#Оглавление)
