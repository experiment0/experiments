from __future__ import annotations
import numpy as np
import pandas as pd 
from typing import Union, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from helpers.criterions import entropy, gini

# тип значения целевой переменной
Value_type = Union[int]




class Node:
    def __init__(
        self, 
        left: Optional[Node]=None, 
        right: Optional[Node]=None, 
        value: Optional[Value_type]=None, 
        log_reg_model: Optional[LogisticRegression]=None, 
        prediction: Optional[np.ndarray]=None,
        impurity: Optional[float]=None,
        weighted_impurity: Optional[float]=None,
        samples: Optional[int]=None, 
        is_leaf: bool=False,
        X: pd.DataFrame = None,
        y: pd.Series = None,
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
            log_reg_model (Optional[LogisticRegression], optional): Обученная модель для разбиения выборки. 
                По умолчанию None.
            prediction (Optional[np.ndarray], optional): Предсказания модели логистической регресси
                для выборки из данной вершины. Предсказания используются как предикат для разделения
                вершины на левую и правую часть.
                По умолчанию None.
            impurity (Optional[float], optional): Значение неоднородности в вершине. 
                По умолчанию None.
            weighted_impurity (Optional[float], optional): Значение взвешенной неоднородности после деления вершины.
                По умолчанию None.
            samples (Optional[int], optional): Количество объектов, попавших в вершину. 
                По умолчанию None.
            is_leaf (bool, optional): Флаг, является ли вершина листовой. 
                По умолчанию False.
            X (pd.DataFrame, optional): Данные выборки в вершине для дальнейших исследований.
                По умолчанию None.
            y: (pd.Series, optional): Истинные значения в вершине для дальнейших исследований.
        """
        self.left = left
        self.right = right
        self.log_reg_model = log_reg_model
        self.prediction = prediction
        self.value = value
        self.impurity = impurity
        self.weighted_impurity = weighted_impurity
        self.samples = samples
        self.is_leaf = is_leaf
        self.X = X
        self.y = y
        
    
    def print(self):
        """Печатает данные вершины (метод для отладки)
        """
        print(
            'Node:',
            f'is_leaf: {self.is_leaf};',
            f'value: {self.value};',
            f'samples: {self.samples};',
            f'impurity: {self.impurity}',
            f'weighted_impurity: {self.weighted_impurity}',
        )




class DecisionTreeWithLogisticRegression:
    def __init__(
        self, 
        criterion: str = 'entropy', 
        max_depth: Optional[int]=None
    ) -> None:
        """Реализует методы алгоритма дерева решений для бинарной классификации.
           В качестве условия для разделения выборки на каждом шаге берется не предикат,
           а логистическая регрессия.

        Args:
            criterion (str, optional): название функции, по которой рассчитывается критерий информативности. 
                По умолчанию 'entropy'.
            max_depth (Optional[int], optional): максимальная глубина дерева решений. 
                По умолчанию None.
        """
        self.max_depth = max_depth
        
        if (criterion == 'entropy'):
            self.criterion = entropy
        elif (criterion == 'gini'):
            self.criterion = gini
        
        # далее в этом поле будет обученное дерево решений
        self.decision_tree: Node = None
    
    
    def __split(self, X: pd.DataFrame, y: pd.Series, prediction: np.ndarray) -> Tuple[
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
    ]:
        """Разделяет выборку на 2 части

        Args:
            X (pd.DataFrame): данные выборки
            y (pd.Series): значения целевой переменной
            prediction (np.ndarray): предсказание для выборки

        Returns:
            Tuple[ pd.DataFrame, pd.Series, pd.DataFrame, pd.Series ]: левая и правая части переданной выборки
        """
        # определяем предикат для каждого значения выборки
        predicat = prediction == 0
        # выделяем левую часть выборки (для которой предикат выполняется)
        X_left, y_left = X[predicat], y[predicat]
        # выделяем правую часть выборки (дополнение к левой части, для которой предикат не выполняется)
        X_right, y_right = X[~predicat], y[~predicat]
        
        return X_left.reset_index().drop(['index'], axis = 1), \
            y_left.reset_index().drop(['index'], axis = 1).iloc[:,0], \
            X_right.reset_index().drop(['index'], axis = 1), \
            y_right.reset_index().drop(['index'], axis = 1).iloc[:,0]
    
    
    def __calculate_weighted_impurity(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        prediction: np.ndarray, 
    ) -> float:
        """Считает взвешенную неоднородность после разбиения выборки

        Args:
            X (pd.DataFrame): данные выборки
            y (pd.Series): значения целевой переменной
            prediction (np.ndarray): предсказание, которое используем как предикат для разделения выборки

        Returns:
            float: взвешенная неоднородность после разделения выборки по предикату
        """
        # разделяем выборку по предикату
        X_left, y_left, X_right, y_right = self.__split(X, y, prediction)
        # определяем размер исходной выборки, а также размеры левой и правой выборок после разделения
        N, N_left, N_right  = y.size, y_left.size, y_right.size
        # считаем значение взвешенной реоднородности
        score = N_left / N * self.criterion(y_left) + N_right / N * self.criterion(y_right)
        
        return score
    
    
    def __get_log_reg_model_and_prediction(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[LogisticRegression, np.ndarray]:
        """Возвращает обученную модель логистической регрессии
               и предсказания для данной выборки.

        Args:
            X (pd.DataFrame): данные обучающей выборки
            y (pd.Series): истинные значения целевой переменной для выборки

        Returns:
            Tuple[LogisticRegression, np.ndarray]: 
                обученная модель логистической регрессии и массив предсказаний
        """
        # создаём объект класса LogisticRegression
        log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
        # обучаем модель
        log_reg_model.fit(X, y)
        # делаем предсказание
        prediction = log_reg_model.predict(X)
        
        return log_reg_model, prediction
    
    
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
            # то это значит, что все элементы в вершине относятся к одному типу 
            return (self.criterion(y) == 0) 
        else:
            return (self.criterion(y) == 0) or (depth > self.max_depth)
        
    
    def __create_leaf_prediction(self, y: pd.Series) -> Value_type:
        """Возвращает предсказание для выборки из листа дерева

        Args:
            y (pd.Series): истинные значения целевой переменной для листа дерева

        Returns:
            Value_type: общее предсказание для листа
        """
        # для классификации предсказание по выборке из листа - это мода
        value = y.mode()[0]
        return value
    
    
    def __build_decision_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> Node:
        """Реализует рекурсивный алгоритм построения дерева решений для бинарной классификации

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
        
        # если в листе нет элементов, сделаем пустую вершину
        if (y.size == 0):
            node = None        
        # если выполняется критерий остановки деления дерева, формируем лист
        elif self.__stopping_criterion(X, y, depth):
            # считаем предсказание для листа
            value = self.__create_leaf_prediction(y)            
                      
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
            # обучаем логистическую регрессию и формируем предсказание для данной вершины
            log_reg_model, prediction = self.__get_log_reg_model_and_prediction(X, y)
            # разделяем выборку по обученной логистической регрессии
            X_left, y_left, X_right, y_right = self.__split(X, y, prediction)
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
                # обученная модель логистической регрессии
                log_reg_model=log_reg_model, 
                # предсказания, полученные по выборке из вершины с помощью обученной логистической регрессии
                prediction=prediction,
                # значение критерия информативности для данной вершины
                impurity=self.criterion(y), 
                # значение взвешенной неоднородности после деления вершины
                weighted_impurity=self.__calculate_weighted_impurity(X, y, prediction),
                # количество элементов в вершине
                samples=y.size,
                X=X,
                y=y,
            )
        
        # возвращаем сформированную вершину,
        # это либо лист, либо корневая или промежуточная вершина с левым и правым потомками
        return node
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Строит и запоминает дерево решений бинарной классификации для обучающей выборки

        Args:
            X (pd.DataFrame): данные обучающей выборки
            y (pd.Series): истинные значения целевой переменной для обучающей выборки
        """
        if (type(X) == np.ndarray):
            X = pd.DataFrame(X)
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
        indent = ('|' + ' '*18) * (depth-1)
        
        # если вершина пустая (формируем такую, если в нее попало 0 элементов)
        if (node is None):
            print(
                indent,
                '|--> ',
                'None',
                sep=''
            )
        # если вершина является листом
        elif node.is_leaf:
            # выводим предсказание для данного листа
            print(
                indent, 
                '|--> ', f'value: {node.value}; samples: {node.samples}; impurity: {node.impurity:.3f}', 
                sep=''
            )
            # если вершина является коневой или внутренней
        else:
            # рекурсивно вызываем текущую функцию для печати данных о левом потомке
            self.__print_decision_tree(node.left, depth=depth)
            
            # выводим параметры потомка данной вершины
            print(indent, '|    ', f'samples: {node.samples}', sep='')
            print(indent, '|--> ', f'impurity: {node.impurity:.3f}', sep='')
            print(indent, '|    ', f'predict: {self.__create_leaf_prediction(pd.Series(node.prediction))}', sep='')
            
            # рекурсивно вызываем текущую функцию для печати данных о правом потомке
            self.__print_decision_tree(node.right, depth=depth)
    
    
    def print_decision_tree(self):
        """Публичный метод для печати параметров дерева решений
        """
        self.__print_decision_tree(self.decision_tree)
    
    
    def __predict_sample(self, node: Node, x: list) -> Value_type:
        """Возвращает предсказание для строки из выборки по данным ранее обученного дерева

        Args:
            node (Node): вершина дерева (при первом вызове функции - корневая вершина)
            x (list): строка из выборки, для которой нужно предсказать значение

        Returns:
            Value_type: предсказание для переданной строки из выборки
        """
        # если вершина является листом
        if (node.is_leaf):
            # вернем значение, предсказанное для данного листа при обучении
            return node.value
        
        # считаем предсказание для переданной строки в текущей вершине
        predict = node.log_reg_model.predict(x)[0]
        
        # если условие предиката для данной вершины выполняется
        if predict == 0:
            # если левый потомок является пустым, вернем для него предсказание родителя
            if (node.left is None):
                return 0
            else:
                # рекурсивно вызываем данную функцию для левого потомка
                return self.__predict_sample(node.left, x)
            # если условие предиката для данной вершины НЕ выполняется,
        else:
            # если правый потомок является пустым, вернем для него предсказания родителя
            if (node.right is None):
                return 1
            else:
                # рекурсивно вызываем данную функцию для правого потомка
                return self.__predict_sample(node.right, x)

    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Формирует предсказания для переданной выборки и ранее обученного дерева решений

        Args:
            X (pd.DataFrame): выборка, для которой нужно сформировать предсказания

        Returns:
            np.ndarray: предсказания для переданной выборки
        """
        if (type(X) == np.ndarray):
            X = pd.DataFrame(X)
        # формируем список предсказаний для каждой строки из выборки
        predictions = [self.__predict_sample(self.decision_tree, [row.values]) for index, row in X.iterrows()]
        # возвращаем массив для предсказаний
        return np.array(predictions)