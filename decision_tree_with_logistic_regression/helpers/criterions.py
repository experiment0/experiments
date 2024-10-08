import pandas as pd
import numpy as np


def squared_error(y: pd.Series) -> float:
        """Считаем среднеквадратичную ошибку для предсказания по переданной выборке

        Args:
            y (pd.Series): истинные значения выборки

        Returns:
            float: посчитанная среднеквадратичная ошибка, если предсказанием является среднее по выборке
        """
        # для значений, попавших в выборку, считаем предсказание как среднее
        y_pred = y.mean()
        # считаем средне квадратичную ошибку для нашего предсказания (MSE)
        return ((y - y_pred) ** 2).mean()


def entropy(y: pd.Series) -> float:
    """Рассчитывает энтропию Шеннона

    Args:
        y (pd.Series): истинные значения выборки

    Returns:
        float: значение энтропии Шеннона
    """
    # считаем доли каждого из классов 
    # (при параметре normalize=True они же являются 
    # вероятностями принадлежности значения к определенному классу)
    p = y.value_counts(normalize=True)
    # суммируем произведения вероятностей на их логарифмы по основанию 2
    entropy = -np.sum(p * np.log2(p))
    return entropy


def gini(y: pd.Series) -> float:
    """Рассчитывает критерий информативности Джинни

    Args:
        y (pd.Series): истинные значения выборки

    Returns:
        float: критерий информативности Джинни
    """
    # считаем доли каждого из классов 
    # (при параметре normalize=True они же являются 
    # вероятностями принадлежности значения к определенному классу)
    p = y.value_counts(normalize=True)
    # суммируем произведения
    gini = np.sum(p * (1 - p))
    return gini