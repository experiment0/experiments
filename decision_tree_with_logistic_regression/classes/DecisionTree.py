from __future__ import annotations
import numpy as np
import pandas as pd 
from typing import Union, Optional, Callable, Tuple
from helpers.criterions import squared_error, entropy, gini



# Тип значения категориальной целевой переменной (для классификации)
Value_categorial_type = Union[str, int]
# Тип значения непрерывной целевой переменной (для регрессии)
Value_numerical_type = float
# Общий тип значения целевой переменной
Value_type = Union[Value_categorial_type, Value_numerical_type]

# Тип для параметров разделения выборки
# первое значение в кортеже - это номер признака
# второе значения в кортеже - это значение признака, по которому нужно разбить выборку на 2 части
# (одна часть - меньше или равно данного значения, другая - больше данного значения)
Split_params_type = tuple[int, float]

# Тип функции для подсчета критерия информативности
Criterion_type = Callable[[pd.Series], float]




class Node:
    def __init__(
        self, 
        left: Optional[Node]=None, 
        right: Optional[Node]=None, 
        value: Optional[Value_type]=None, 
        split_params: Optional[Split_params_type]=None, 
        impurity: Optional[float]=None,
        weighted_impurity: Optional[float]=None,
        samples: Optional[int]=None, 
        is_leaf: bool=False
    ):
        """Вспомогательный класс вершины дерева решений

        Args:
            left (Optional[Node], optional): Ссылка на левого потомка. 
                По умолчанию None.
            right (Optional[Node], optional): Ссылка на правого потомка. 
                По умолчанию None.
            value (Optional[Value_type], optional): Предсказание для данной вершины 
                (устанавливается, если вершина является листом). 
                По умолчанию None.
            split_params (Optional[Split_params_type], optional): Параметры разбиения выборки. 
                По умолчанию None.
            impurity (Optional[float], optional): Значение неоднородности в вершине. 
                По умолчанию None.
            weighted_impurity (Optional[float], optional): Значение взвешенной неоднородности после деления вершины.
                По умолчанию None.
            samples (Optional[int], optional): Количество объектов, попавших в вершину. 
                По умолчанию None.
            is_leaf (bool, optional): Флаг, является ли вершина листовой. 
                По умолчанию False.
        """
        self.left = left
        self.right = right
        self.split_params = split_params
        self.value = value
        self.impurity = impurity
        self.weighted_impurity = weighted_impurity
        self.samples = samples
        self.is_leaf = is_leaf




class DecisionTree:
    def __init__(self, max_depth: Optional[int]=None) -> None:
        """Реализует основные методы алгоритма дерева решений.

        Args:
            max_depth (Optional[int], optional): максимальная глубина дерева решений. 
                                                 По умолчанию None.
        """
        self.max_depth = max_depth
        
        # далее в этом поле будет обученное дерево решений
        self.decision_tree: Node = None
        # далее в этом поле будет количество признаков обучающей выборки
        self.features_count: int = 0
        
    
    
    def __find_candidates_for_thresholds(self, x: pd.Series, y: pd.Series) -> list:
        """Формирует и возвращает список значений признака, для каждого из которых
        имеет смысл сделать разделение выборки на 2 части (больше и меньше значения признака).

        Args:
            x (pd.Series): столбец с рассматриваемым признаком
            y (pd.Series): столбец со значениями целевой переменной

        Returns:
            list: список значений признака для разделения выборки
        """
        # отсортируем данные по возрастанию и удалим дубликаты
        x = x.sort_values().drop_duplicates()
        # вычислим скользящее среднее между каждыми двумя соседними значениями признака
        # первым в списке будет nan, поэтому для его удаления воспользуемся методом dropna
        x_roll_mean = x.rolling(2).mean().dropna()
        # возьмем значения целевой переменной для отсортированных и отобранных индексов
        y = y[x_roll_mean.index]
        # подсчитаем разницу между соседними значениями y
        y_diff = y.diff()
        # оставим только те значения x, при переходе через которые значение y меняется
        candidates = x_roll_mean[y_diff != 0]
        
        # вернем список значений, для которых имеет смысл делать проверку
        return candidates.values
    
    
    def __split(self, X: pd.DataFrame, y: pd.Series, split_params: Split_params_type) -> Tuple[
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
    ]:
        """Разделяет выборку на 2 части по предикату.
        В левую часть идут значения, для которых предикат выполняется.
        В правую - для которых предикат не выполняется.

        Args:
            X (pd.DataFrame): строки выборки
            y (pd.Series): значение целевого признака выборки
            split_params (Split_params_type): параметры для разделения выборки (номер и значение признака)

        Returns:
            Tuple[ pd.DataFrame, pd.Series, pd.DataFrame, pd.Series ]: левая и правая части переданной выборки
        """
        # вынимаем номер признака и значение предиката
        j, t = split_params
        # формируем маску для разделения выборки по предикату
        predicat = X.iloc[:, j] <= t
        # выделяем левую часть выборки (для которой предикат выполняется)
        X_left, y_left = X[predicat], y[predicat]
        # выделяем правую часть выборки (дополнение к левой части), для которой предикат не выполняется
        X_right, y_right = X[~predicat], y[~predicat]
        
        return X_left, y_left, X_right, y_right
    
    
    def __calculate_weighted_impurity(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        split_params: Split_params_type, 
    ) -> float:
        """Считает взвешенную неоднородность после разбиения выборки

        Args:
            X (pd.DataFrame): данные выборки
            y (pd.Series): значения целевой переменной
            split_params (Split_params_type): параметры разделения 
                                              (номер признака и его значение для разбиения)

        Returns:
            float: взвешенная неоднородность после разделения выборки по предикату
        """
        # разделяем выборку по предикату
        X_left, y_left, X_right, y_right = self.__split(X, y, split_params)
        # определяем размер исходной выборки, а также размеры левой и правой выборок после разделения
        N, N_left, N_right  = y.size, y_left.size, y_right.size
        # считаем значение взвешенной реоднородности
        score = N_left / N * self.criterion(y_left) + N_right / N * self.criterion(y_right)
        
        return score
    
    
    def __get_best_split_params(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
    ) -> Split_params_type:
        """Определяет и возвращает лучшие параметры для разделения выборки

        Args:
            X (pd.DataFrame): данные выборки
            y (pd.Series): значения целевой переменной

        Returns:
            Split_params_type: лучшие параметры для разделения выборки
        """
        # количество признаков в выборке
        features_count = X.shape[1]
        # значение взвешенной неоднородности после разделения выборки
        # (сначала берем бесконечность, дальше будем искать, с какими параметрами получим минимальное)
        min_weighted_impurity = np.inf
        # лучшие параметры для разделения выборки 
        # (такие, при которых взвешенная неоднородность после разделения будет минимальна)
        best_split_params = None
        
        # итерируемся по каждому признаку в выборке
        for j in range(features_count):
            # получаем список значений признака для разделения выборки
            thresholds = self.__find_candidates_for_thresholds(X.iloc[:, j], y)
            # итерируемся по каждому значению признака
            for t in thresholds:
                # параметры для разделения выборки (номер признака и его значение)
                split_params = (j, t)
                # считаем взвешенную неоднородность после разбиения выборки
                weighted_impurity = self.__calculate_weighted_impurity(X, y, split_params)
                # если значение взвешенной неоднородности меньше, чем предыдущее минимальное
                if weighted_impurity < min_weighted_impurity:
                    # запоминаем новое минимальное значение
                    min_weighted_impurity = weighted_impurity
                    # и запоминаем параметры разделения выборки как лучшие
                    best_split_params = split_params
        
        # возвращаем параметры разделения выборки, 
        # для которых значение взвешенной неоднородноти оказалось минимальным
        return best_split_params

    
    def __stopping_criterion(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        depth: int = 0,
    ) -> bool:
        """Критерий остановки деления дерева

        Args:
            X (pd.DataFrame): данные выборки
            y (pd.Series): истинные значения целевой переменной
            depth (int): текущая глубина дерева

        Returns:
            bool: флаг, нужно ли остановить деление дерева для данной вершины
        """
        # если не задана максимально допустимая глубина дерева
        if self.max_depth is None:
            # если критерий информативности в вершине равен 0,
            # то это значит, что все элементы в вершине относятся к одному типу (для классификации)
            # или что дисперсия (разброс) равен 0 (для регрессии)
            return (self.criterion(y) == 0) 
        else:
            return (self.criterion(y) == 0) or (depth > self.max_depth)
    
    
    def __build_decision_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> Node:
        """Реализует рекурсивный алгоритм построения дерева решений

        Args:
            X (pd.DataFrame): данные выборки
            y (pd.Series): истинные значения целевой переменной
            depth (int): текущая глубина дерева. По умолчанию 0.

        Returns:
            Node: корневая вершина дерева, которая ссылается на левого и правого потомков,
                  они в свою очередь ссылаются на своих потомков,
                  и так до самых листьев.
        """
        # увеличиваем значение текущей глубины дерева
        depth += 1
        
        # если выполняется критерий остановки деления дерева, формируем лист
        if self.__stopping_criterion(X, y, depth):
            # считаем предсказание для листа 
            # (данный метод реализован отдельно для дочерних классов 
            # дерева регрессии и дерева классификации, 
            # здесь в родительском классе его нет)
            value = self.create_leaf_prediction(y)
            # формируем объект класса вершины дерева для листа
            node = Node(
                # предсказание для листа
                value=value,
                # значение критерия информативности для выборки из листа
                impurity=self.criterion(y), 
                # количество объектов в листе
                samples=y.size,
                # флаг, что вершина является листом
                is_leaf=True
            )
            # если НЕ выполняется критерий остановки деления дерева, производим дальнейшее деление
        else:
            # находим наилучшие параметры для разбиения выборки (номер признака и его значение)
            best_split_params = self.__get_best_split_params(X, y)
            # получаем разделение выборки на левую (где условие предиката выполняется)
            # и правую (где условие предиката не выполняется)
            X_left, y_left, X_right, y_right = self.__split(X, y, best_split_params)
            # вызываем данную функцию рекурсивно и формируем левую дочернюю вершину
            left_node = self.__build_decision_tree(X_left, y_left, depth)
            # вызываем данную функцию рекурсивно и формируем правую дочернюю вершину
            right_node = self.__build_decision_tree(X_right, y_right, depth)
            # формируем корневую или промежуточную вершину с левым и правым потомками
            node = Node(
                # вершина левого потомка
                left=left_node, 
                # вершина правого потомка
                right=right_node, 
                # параметры разделения данной вершины
                split_params=best_split_params, 
                # значение критерия информативности для данной вершины
                impurity=self.criterion(y), 
                # значение взвешенной неоднородности после деления вершины
                weighted_impurity=self.__calculate_weighted_impurity(X, y, best_split_params),
                # количество элементов в вершине
                samples=y.size
            )
        
        # возвращаем сформированную вершину,
        # это либо лист, либо корневая или промежуточная вершина с левым и правым потомками
        return node
        
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Строит и запоминает дерево решений для обучающей выборки

        Args:
            X (pd.DataFrame): данные обучающей выборки
            y (pd.Series): истинные значения целевой переменной для обучающей выборки
        """
        # запоминаем количество признаков в обучающей выборке
        self.features_count = X.shape[1]
        
        # строим дерево решений
        decision_tree = self.__build_decision_tree(X, y)
        # запоминаем его для дальнейших предсказаний
        self.decision_tree = decision_tree
    
    
    def __print_decision_tree(self, node: Node, depth: int = 0) -> None:
        """Печатает данные о вершинах дерева решений

        Args:
            node (Node): вершина дерева
            depth (int, optional): текущая глубина дерева. 
                                   По умолчанию 0.
        """
        # прибавляем единицу к текущей глубине дерева
        depth += 1
        # строка с отступом для текущего уровня дерева
        indent = '|   '*(depth-1) + '|---'
        
        # если вершина является листом
        if node.is_leaf:
            # выводим предсказание для данного листа
            print(indent, 'value: [{}]'.format(node.value))
            # если вершина является коневой или внутренней
        else:
            # выводим параметры предиката для левого потомка данной вершины
            print(indent, 'feature_{} <= {:.2f}'.format(*node.split_params))
            # рекурсивно вызываем текущую функцию для печати данных о левом потомке
            self.__print_decision_tree(node.left, depth=depth)
            
            # выводим параметры предиката для правого потомка данной вершины
            print(indent, 'feature_{} > {:.2f}'.format(*node.split_params))
            # рекурсивно вызываем текущую функцию для печати данных о правом потомке
            self.__print_decision_tree(node.right, depth=depth)
    
    
    def print_decision_tree(self):
        """Публичный метод для печати параметров дерева решений
        """
        self.__print_decision_tree(self.decision_tree)
            

    def __predict_sample(self, node: Node, x: np.ndarray) -> Value_type:
        """Возвращает предсказание для строки из выборки по данным ранее обученного дерева

        Args:
            node (Node): вершина дерева (при первом вызове функции - корневая вершина)
            x (np.ndarray): строка из выборки, для которой нужно предсказать значение

        Returns:
            Value_type: предсказание для переданной строки из выборки
        """
        # если вершина является листом
        if node.is_leaf:
            # вернем значение, предсказанное для данного листа при обучении
            return node.value
        # параметры разбиения вершины
        # (номер и значение признака)
        j, t = node.split_params
        # если условие предиката для данной вершины выполняется
        if x[j] <= t:
            # рекурсивно вызываем данную функцию для левого потомка
            return self.__predict_sample(node.left, x)
        else:
            # если условие предиката для данной вершины НЕ выполняется,
            # рекурсивно вызываем данную функцию для правого потомка
            return self.__predict_sample(node.right, x)

    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Формирует предсказания для переданной выборки и ранее обученного дерева решений

        Args:
            X (pd.DataFrame): выборка, для которой нужно сформировать предсказания

        Returns:
            np.ndarray: предсказания для переданной выборки
        """
        # формируем список предсказаний для каждой строки из выборки
        predictions = [self.__predict_sample(self.decision_tree, x) for x in X.values]
        # возвращаем массив для предсказаний
        return np.array(predictions)
    



class DecisionTreeRegressor(DecisionTree):
    def __init__(self, criterion: str = 'squared_error', **kwargs) -> None:
        """Класс для построения дерева решений регрессии.

        Args:
            criterion (str, optional): название функции, по которой рассчитывается критерий информативности. 
                По умолчанию 'squared_error'.
        """
        # вызываем конструктор родительского класса, чтобы инициализировать остальные атрибуты
        DecisionTree.__init__(self, **kwargs)
        
        if (criterion == 'squared_error'):
            self.criterion = squared_error
    
    
    def create_leaf_prediction(self, y: pd.Series) -> Value_numerical_type:
        """Возвращает предсказание для выборки из листа дерева

        Args:
            y (pd.Series): истинные значения целевой переменной для листа дерева

        Returns:
            Value_numerical_type: общее предсказание для листа
        """
        # для регрессии предсказание по выборке из листа - это среднее
        value = y.mean()
        return value




class DecisionTreeClassifier(DecisionTree):
    def __init__(self, criterion: str = 'entropy', **kwargs) -> None:
        """Класс для построения дерева решений классификации

        Args:
            criterion (str, optional): название функции, по которой рассчитывается критерий информативности. 
                По умолчанию 'entropy'.
        """
        # вызываем конструктор родительского класса, чтобы инициализировать остальные атрибуты
        DecisionTree.__init__(self, **kwargs)
        
        if (criterion == 'entropy'):
            self.criterion = entropy
        elif (criterion == 'gini'):
            self.criterion = gini
    
    
    def create_leaf_prediction(self, y: pd.Series) -> Value_categorial_type:
        """Возвращает предсказание для выборки из листа дерева

        Args:
            y (pd.Series): истинные значения целевой переменной для листа дерева

        Returns:
            Value_categorial_type: общее предсказание для листа
        """
        # для классификации предсказание по выборке из листа - это мода
        value = y.mode()[0]
        return value
    


