import numpy as np
import sympy
from sympy import *
import matplotlib.pyplot as plt 
from classes.FSM import FSM, State, Action


class GradientDescent:
    def __init__(
        self, 
        function: sympy.core.add.Add,
        x: sympy.core.symbol.Symbol,
        y: sympy.core.symbol.Symbol,
        point_start: np.ndarray = np.array([1, 1]),
        step_size: float = 0.01,
        steps_count: int = 100,
        min_acceptable_gradient: float = 0.001,
        graph_size: int = 6,
        graph_points_count: int = 100,
        should_display_logs: bool = false,
    ) -> None:
        """Реализует алгоритм градиентного спуска для функции 2-х переменных и выводит график

        Args:
            function (sympy.core.add.Add): функция
            x (sympy.core.symbol.Symbol): переменная x
            y (sympy.core.symbol.Symbol): переменная y
            point_start (np.ndarray): координаты начальной точки
            step_size (float): величина шага
            steps_count (int): количество шагов
            min_acceptable_gradient (float): минимальное значение частной производной, 
                на котором алгоритм поиска точки минимума останавливается
            graph_size (int): диапазон значений графика по x и y
            graph_points_count (int): количесво точек графика в заданном диапазоне            
            should_display_logs (bool): нужно ли выводить логи
        """
        self.function = function
        self.x = x
        self.y = y
        self.get_function_value = lambdify([x, y], function, 'numpy')
        
        self.point_start = point_start
        self.step_size = step_size
        self.steps_count = steps_count
        self.min_acceptable_gradient = min_acceptable_gradient
        # список с найденными точками градиентного спуска
        self.gradient_points = [point_start]
        
        self.graph_radius = graph_size/2
        self.graph_points_count = graph_points_count
        
        self.should_display_logs = should_display_logs
        
        # объект класса FSM для определения текущего состояния программы и ее следующего действия
        self.fsm = FSM(should_display_logs)
        
    
    def get_gradient(self, x: float, y: float) -> np.ndarray:
        """Возвращает вектор градиента функции в точке (x, y)

        Args:
            x (float): координата по x
            y (float): координата по y

        Returns:
            np.ndarray: вектор градиента
        """
        # частная производная по x
        f_x = diff(self.function, self.x)
        # частная производная по y
        f_y = diff(self.function, self.y)
        
        return np.array([
            # значение частной производной по x в переданной точке
            f_x.evalf(subs={self.x: x, self.y: y}),
            # значение частной производной по y в переданной точке
            f_y.evalf(subs={self.x: x, self.y: y}),
        ])
    
    
    def get_step_last(self) -> int:
        """Возвращает номер шага, на котором остановились

        Returns:
            int: номер шага
        """
        return len(self.gradient_points) - 1
    
    
    def set_next_gradient_point(self) -> None:
        """Находит и устанавливает следующую точку градиентного спуска
        """
        # последняя точка, которую нашли
        point_current = self.gradient_points[-1]
        # определяем новую точку: из текущей вычитаем вектор градиента в ней, умноженный на шаг градиента
        point_new = point_current - self.step_size * self.get_gradient(point_current[0], point_current[1])
        # добавляем в список новую точку
        self.gradient_points.append(point_new)
        
    
    def run_fsm_logic(self) -> None:
        """Выполняет логику класса FSM для определения текущего состояния программы и ее следующего действия
        """
        # определяем текущее состояние
        self.detect_fsm_state()
        # выполняем следующее действие, которое соответствует текущему состоянию         
        self.run_fsm_action()
        
    
    def detect_fsm_state(self) -> None:
        """Определяет текущее состояние программы
        """
        # текущая найденная точка градиентного спуска
        point_current = self.gradient_points[-1]
        # предыдущая точка градиентного спуска
        point_before = self.gradient_points[-2]
        
        # значение функции в текущей точке
        function_value_current = self.get_function_value(point_current[0], point_current[1])
        # значение функции в предыдущей точке
        function_value_before = self.get_function_value(point_before[0], point_before[1])
        # разница значений в текущей и предыдущей точках
        function_difference = function_value_current - function_value_before
        
        # градиент в текущей точке
        current_gradient = self.get_gradient(point_current[0], point_current[1])        

        # номер текущего шага
        step_last = self.get_step_last()
        
        # если значение функции увеличилось
        if (function_difference > 0):
            self.fsm.set_state(State.DETECT_FUNCTION_GROWTHING.value)
        elif (
            # если мы исчерпали количество шагов, 
            # но желаемого значения частных производных в точке псевдоминимума не достигли
            step_last >= self.steps_count and 
            (abs(current_gradient[0]) > self.min_acceptable_gradient or 
             abs(current_gradient[1]) > self.min_acceptable_gradient)
         ):
            self.fsm.set_state(State.STEPS_END_UNTIL_SUCCESS.value)
        else:
            self.fsm.set_state(State.READY_TO_FIND_NEXT_POINT.value)
            
    
    def run_fsm_action(self) -> None:
        """Выполняет действие, определенное конфигом FSM для текущего состояния
        """
        if (self.fsm.next_action == Action.RETURN_TO_PREVIOUS_POINT_WITH_DESCRASING_STEP.value):
            self.return_to_previous_point_with_decreasing_step()
            self.fsm.set_next_state()
        elif (self.fsm.next_action == Action.ADD_STEPS.value):
            self.add_steps()
            self.fsm.set_next_state()
        elif (self.fsm.next_action == Action.FIND_NEXT_POINT.value):
            pass
            
    
    def return_to_previous_point_with_decreasing_step(self) -> None:
        """Возвращается к предыдущей точке и уменьшает шаг градиента вдвое
        """
        self.step_size = self.step_size / 2
        self.gradient_points.pop(-1)
        
    
    def add_steps(self) -> None:
        """Добавляет еще 10 шагов
        """
        self.steps_count = self.steps_count + 10
        
    
    def set_gradient_points(self) -> None:
        """Запускает цикл по поиску и установлению точек градиентного спуска
        """
        # номер текущего шага
        step_last = self.get_step_last()
        
        # пока мы не исчерпали количество шагов
        while step_last < self.steps_count:
            # находим и устанавливаем следующую точку градиента
            self.set_next_gradient_point()
            # определяем текущее состояние программы и выполняем следующее по конфигу FSM действие
            self.run_fsm_logic()
            # снова определяем текущий шаг
            step_last = self.get_step_last()
    
        
    def display_logs(self, step: int, point_current: np.ndarray) -> None:
        """Выводит логи расчета координат градиентного спуска

        Args:
            step (int): номер шага
            point_current (np.ndarray): координаты текущей точки
        """
        x = point_current[0]
        y = point_current[1]
        gradient = self.get_gradient(x, y)
        
        print(f'Номер шага: {step}')
        print(f'Координаты точки: x={x:.4f}, y={y:.4f}')
        print(f'Значение функции в точке: {self.get_function_value(x, y):.4f}')
        print(f'Вектор градиента в точке: ({gradient[0]:.4f}, {gradient[1]:.4f})')
        print()
        
    
    def display_graph(self) -> None:
        """Выводит график функции, ее линии уровня,
        а также кривую, построенную на координатах точек градиентного спуска.
        """
        # последняя найденная точка градиентного спуска будет центром графика
        point_center = self.gradient_points[-1]
        x_center = float(point_center[0])
        y_center = float(point_center[1])
        
        # от центральной точки отсчитываем диапазон значений по x и y
        x_min = x_center - self.graph_radius
        x_max = x_center + self.graph_radius
        y_min = y_center - self.graph_radius
        y_max = y_center + self.graph_radius
        
        # формируем набор координат по x
        coordinates_x = np.linspace(x_min, x_max, self.graph_points_count)
        # формируем набор координат по y
        coordinates_y = np.linspace(y_min, y_max, self.graph_points_count)
        
        # формируем сетку координат по x и y
        grid_x, grid_y = np.meshgrid(coordinates_x, coordinates_y)
        # считаем значение функции для каждой точки из сформированной сетки
        grid_z = self.get_function_value(grid_x, grid_y)
        
        # задаем фигуру и координатную плоскость
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # рисуем непрерывный график для сформированной сетки координат по трем осям
        ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.6)
        # рисуем линии уровня
        ax.contour(grid_x, grid_y, grid_z, zdir='z', offset=grid_z.min())
        
        gradient_points_array = np.array(self.gradient_points)
        # все координаты x точек градиентного
        coordinates_gradient_x = gradient_points_array[:, 0]
        # все координаты y точек градиентного
        coordinates_gradient_y = gradient_points_array[:, 1]
        # значения функции в каждой точке градиентного спуска
        coordinates_gradient_z = self.get_function_value(coordinates_gradient_x, coordinates_gradient_y)
        # выводим линию, как мы считали градиентный спуск
        ax.plot(coordinates_gradient_x, coordinates_gradient_y, coordinates_gradient_z, "o-")

        # устанавливаем границы графика по осям
        ax.set_xlim(grid_x.min(), grid_x.max())
        ax.set_ylim(grid_y.min(), grid_y.max())
        ax.set_zlim(grid_z.min(), grid_z.max())
    
    
    def display_result(self) -> None:
        """Выводит результат подсчетов (логи по последней точке)
        """
        print('---------------------------------')
        print('Данные последней найденной точки:')
        # номер последнего шага
        step_last = self.get_step_last()
        # последняя найденная точка
        point_last = self.gradient_points[step_last]
        # выводим логи
        self.display_logs(step_last, point_last)
        

    def run(self) -> None:
        """Запускает процесс расчета координат градиентного спуска, выводит график и результат
        """
        # считаем и запоминаем координаты точек градиентного спуска
        self.set_gradient_points()
        # выводим график
        self.display_graph()
        # выводим результат
        self.display_result()        
            