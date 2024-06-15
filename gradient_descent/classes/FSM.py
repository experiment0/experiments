from enum import Enum

# состояния
class State(Enum):
    # готовность к поиску следующей точки
    READY_TO_FIND_NEXT_POINT = 'ready_to_find_next_point'
    # ожидание программного определения следующего состояния
    WAITING_OF_DETECT_STATE_BY_PROGRAMM = 'waiting_of_detect_state_by_programm'
    # обнаружен рост функции
    DETECT_FUNCTION_GROWTHING = 'detect_function_growthing'
    # шаги закончились до достижения успешного результата 
    # (успешный результат определяется величиной частных производных в текущей точке)
    STEPS_END_UNTIL_SUCCESS = 'steps_end_until_success'
    
    
# действия
class Action(Enum):
    # возврат к предыдущей точке с уменьшением шага градиента (множителя)
    RETURN_TO_PREVIOUS_POINT_WITH_DESCRASING_STEP = 'return_to_previous_point_with_decreasing_step'
    # поиск координат следующей точки
    FIND_NEXT_POINT = 'find_next_point'
    # определение следующего состояния с помощью программы
    DETECT_STATE_USING_BY_PROGRAMM = 'detect_state_using_by_programm'
    # добавление дополнительных шагов
    ADD_STEPS = 'add_steps'


# конфиг переходов: текущее_состояние: (действие, новое_состояние)
transition_config = {
    State.READY_TO_FIND_NEXT_POINT.value:
        (Action.FIND_NEXT_POINT.value, State.WAITING_OF_DETECT_STATE_BY_PROGRAMM.value),
    State.WAITING_OF_DETECT_STATE_BY_PROGRAMM.value:
        (Action.DETECT_STATE_USING_BY_PROGRAMM.value, State.WAITING_OF_DETECT_STATE_BY_PROGRAMM.value),
    State.DETECT_FUNCTION_GROWTHING.value:
        (Action.RETURN_TO_PREVIOUS_POINT_WITH_DESCRASING_STEP.value, State.READY_TO_FIND_NEXT_POINT.value),
    State.STEPS_END_UNTIL_SUCCESS.value:
        (Action.ADD_STEPS.value, State.READY_TO_FIND_NEXT_POINT.value)
}


class FSM:
    def __init__(self, should_display_logs: bool) -> None:
        """По принципу паттерна Finite State Mashine (FSM) определяет 3 значения:
        - текущее состояние программы
        - следующее действие
        - состояние, в которое программа перейдет после выполнения данного дейстия

        Args:
            should_display_logs (bool): нужно ли выводить логи
        """
        self.should_display_logs = should_display_logs
        self.current_state: State = State.READY_TO_FIND_NEXT_POINT.value
        self.update()
        
    def update(self) -> None:
        """Обновляет значения полей со следующем действием и следующим состоянием
        """
        # конфиг для текущего состояния
        current_config = transition_config[self.current_state]
        # устанавливаем следующее действие
        self.next_action: Action = current_config[0]
        # устанавливаем состояние, в которое программа перейдет после его выполнения
        self.next_state: State = current_config[1]
        self.display_logs()
        
    def set_state(self, new_state: State) -> None:
        """Устанавливает новое состояние

        Args:
            new_state (State): значение нового состояния
        """
        self.current_state = new_state
        self.update()
        
    def set_next_state(self) -> None:
        """Устанавливает следующее состояние по конфигу (вызывается после выполнения действия)
        """
        self.current_state = self.next_state
        self.update()
    
    def display_logs(self) -> None:
        """Выводит логи
        """
        if (self.should_display_logs):
            print('--- FSM logs ---')
            print('current_state: ', self.current_state)
            print('next_action: ', self.next_action)
            print('next_state: ', self.next_state)
            print('--- / FSM logs ---')
    
    