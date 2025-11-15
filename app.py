import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
import re
import time

# Настройка страницы
st.set_page_config(
    page_title="MatrixLab - Калькулятор матриц",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)


class MatrixCalculator:
    """Класс для выполнения операций с матрицами"""

    def __init__(self):
        self.history = []

    def parse_matrix_input(self, input_text, rows, cols):
        """Парсинг ввода матрицы из текста"""
        try:
            # Удаляем лишние пробелы и разбиваем на строки
            lines = [line.strip() for line in input_text.strip().split('\n') if line.strip()]

            matrix = []
            for line in lines[:rows]:  # Ограничиваем количеством строк
                # Разбиваем строку на числа (разделители: пробелы, запятые, табы)
                numbers = re.split(r'[,\s\t]+', line.strip())
                row = []
                for num in numbers[:cols]:  # Ограничиваем количеством столбцов
                    if num:
                        row.append(float(num))
                if row:
                    matrix.append(row)

            # Проверяем, что все строки имеют одинаковую длину
            if matrix and len(set(len(row) for row in matrix)) != 1:
                st.error("Все строки матрицы должны иметь одинаковое количество элементов!")
                return None

            return np.array(matrix)
        except ValueError as e:
            st.error(f"Ошибка ввода данных: {e}")
            return None

    def create_empty_matrix_input(self, rows, cols, matrix_id):
        """Создает текстовое поле для ввода пустой матрицы"""
        default_text = ""
        for i in range(rows):
            for j in range(cols):
                default_text += "0 "
            default_text += "\n"

        return st.text_area(
            f"Матрица {matrix_id} ({rows}×{cols})",
            value=default_text.strip(),
            height=100,
            key=f"matrix_{matrix_id}"
        )

    def add_to_history(self, operation, matrix1, matrix2=None, result=None):
        """Добавляет операцию в историю"""
        history_item = {
            'operation': operation,
            'matrix1': matrix1.copy() if matrix1 is not None else None,
            'matrix2': matrix2.copy() if matrix2 is not None else None,
            'result': result.copy() if result is not None else None,
            'timestamp': time.time()
        }
        self.history.append(history_item)

    # БАЗОВЫЕ ОПЕРАЦИИ

    def transpose(self, matrix):
        """Транспонирование матрицы"""
        try:
            result = matrix.T
            self.add_to_history('Транспонирование', matrix, None, result)
            return result
        except Exception as e:
            st.error(f"Ошибка при транспонировании: {e}")
            return None

    def multiply_matrices(self, matrix1, matrix2):
        """Умножение матриц"""
        try:
            if matrix1.shape[1] != matrix2.shape[0]:
                st.error("Количество столбцов первой матрицы должно совпадать с количеством строк второй матрицы!")
                return None

            result = np.dot(matrix1, matrix2)
            self.add_to_history('Умножение матриц', matrix1, matrix2, result)
            return result
        except Exception as e:
            st.error(f"Ошибка при умножении матриц: {e}")
            return None

    def add_matrices(self, matrix1, matrix2):
        """Сложение матриц"""
        try:
            if matrix1.shape != matrix2.shape:
                st.error("Матрицы должны быть одного размера для сложения!")
                return None

            result = matrix1 + matrix2
            self.add_to_history('Сложение матриц', matrix1, matrix2, result)
            return result
        except Exception as e:
            st.error(f"Ошибка при сложении матриц: {e}")
            return None

    def subtract_matrices(self, matrix1, matrix2):
        """Вычитание матриц"""
        try:
            if matrix1.shape != matrix2.shape:
                st.error("Матрицы должны быть одного размера для вычитания!")
                return None

            result = matrix1 - matrix2
            self.add_to_history('Вычитание матриц', matrix1, matrix2, result)
            return result
        except Exception as e:
            st.error(f"Ошибка при вычитании матриц: {e}")
            return None

    def scalar_multiply(self, matrix, scalar):
        """Умножение матрицы на скаляр"""
        try:
            result = matrix * scalar
            self.add_to_history('Умножение на скаляр', matrix, None, result)
            return result
        except Exception as e:
            st.error(f"Ошибка при умножении на скаляр: {e}")
            return None

    # АНАЛИТИЧЕСКИЕ ОПЕРАЦИИ

    def determinant(self, matrix):
        """Вычисление определителя матрицы"""
        try:
            if matrix.shape[0] != matrix.shape[1]:
                st.error("Матрица должна быть квадратной для вычисления определителя!")
                return None

            result = np.linalg.det(matrix)
            self.add_to_history('Определитель', matrix, None, np.array([[result]]))
            return result
        except Exception as e:
            st.error(f"Ошибка при вычислении определителя: {e}")
            return None

    def inverse_matrix(self, matrix):
        """Нахождение обратной матрицы"""
        try:
            if matrix.shape[0] != matrix.shape[1]:
                st.error("Матрица должна быть квадратной для нахождения обратной!")
                return None

            det = np.linalg.det(matrix)
            if abs(det) < 1e-10:
                st.error("Матрица вырожденная, обратной матрицы не существует!")
                return None

            result = np.linalg.inv(matrix)
            self.add_to_history('Обратная матрица', matrix, None, result)
            return result
        except Exception as e:
            st.error(f"Ошибка при нахождении обратной матрицы: {e}")
            return None

    def matrix_rank(self, matrix):
        """Вычисление ранга матрицы"""
        try:
            result = np.linalg.matrix_rank(matrix)
            self.add_to_history('Ранг матрицы', matrix, None, np.array([[result]]))
            return result
        except Exception as e:
            st.error(f"Ошибка при вычислении ранга: {e}")
            return None

    # ПРОДВИНУТЫЕ ОПЕРАЦИИ

    def eigenvalues_eigenvectors(self, matrix):
        """Нахождение собственных значений и векторов"""
        try:
            if matrix.shape[0] != matrix.shape[1]:
                st.error("Матрица должна быть квадратной для нахождения собственных значений!")
                return None, None

            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            self.add_to_history('Собственные значения', matrix, None, np.diag(eigenvalues))
            return eigenvalues, eigenvectors
        except Exception as e:
            st.error(f"Ошибка при нахождении собственных значений: {e}")
            return None, None

    def solve_linear_system(self, coefficients, constants):
        """Решение системы линейных уравнений"""
        try:
            if coefficients.shape[0] != constants.shape[0]:
                st.error("Количество уравнений должно совпадать с количеством констант!")
                return None

            result = np.linalg.solve(coefficients, constants)
            self.add_to_history('Решение СЛАУ', coefficients, constants, result)
            return result
        except Exception as e:
            st.error(f"Ошибка при решении системы уравнений: {e}")
            return None


def display_matrix(matrix, title="Матрица"):
    """Красивое отображение матрицы"""
    if matrix is None:
        return

    st.subheader(title)

    # Создаем DataFrame для красивого отображения
    if matrix.ndim == 1:
        # Вектор
        df = pd.DataFrame(matrix.reshape(1, -1))
    elif matrix.shape[0] == 1 and matrix.shape[1] == 1:
        # Скаляр
        st.write(f"**Результат:** {matrix[0, 0]:.6f}")
        return
    else:
        # Матрица
        df = pd.DataFrame(matrix)

    # Отображаем матрицу
    st.dataframe(df.style.format("{:.6f}"), use_container_width=True)

    # Показываем размерность
    st.caption(f"Размерность: {matrix.shape}")


def main():
    """Основная функция приложения"""

    # Инициализация калькулятора
    if 'calculator' not in st.session_state:
        st.session_state.calculator = MatrixCalculator()

    calculator = st.session_state.calculator

    # Заголовок приложения
    st.title("🧮 MatrixLab")
    st.markdown("### Мощный калькулятор матриц для студентов и профессионалов")

    # Боковая панель для навигации
    st.sidebar.title("Навигация")
    app_mode = st.sidebar.selectbox(
        "Выберите раздел",
        ["Калькулятор", "История вычислений", "Справка"]
    )

    if app_mode == "Калькулятор":
        render_calculator(calculator)
    elif app_mode == "История вычислений":
        render_history(calculator)
    else:
        render_help()


def render_calculator(calculator):
    """Рендер основной калькуляторной части"""

    st.sidebar.header("Настройки матриц")

    # Выбор размеров матриц
    col1, col2 = st.sidebar.columns(2)
    with col1:
        rows1 = st.number_input("Строки матрицы A", min_value=1, max_value=10, value=2)
        cols1 = st.number_input("Столбцы матрицы A", min_value=1, max_value=10, value=2)
    with col2:
        rows2 = st.number_input("Строки матрицы B", min_value=1, max_value=10, value=2)
        cols2 = st.number_input("Столбцы матрицы B", min_value=1, max_value=10, value=2)

    # Выбор операции
    operation = st.sidebar.selectbox(
        "Выберите операцию",
        [
            "Транспонирование",
            "Умножение матриц",
            "Сложение матриц",
            "Вычитание матриц",
            "Умножение на скаляр",
            "Определитель",
            "Обратная матрица",
            "Ранг матрицы",
            "Собственные значения",
            "Решение СЛАУ"
        ]
    )

    # Основные колонки для ввода матриц
    col1, col2 = st.columns(2)

    with col1:
        st.header("Матрица A")
        matrix_a_input = calculator.create_empty_matrix_input(rows1, cols1, "A")
        matrix_a = calculator.parse_matrix_input(matrix_a_input, rows1, cols1)

        if matrix_a is not None:
            display_matrix(matrix_a, "Матрица A")

    with col2:
        # Для операций с одной матрицей скрываем вторую матрицу
        if operation not in ["Транспонирование", "Определитель", "Обратная матрица",
                             "Ранг матрицы", "Собственные значения", "Умножение на скаляр"]:
            st.header("Матрица B")
            matrix_b_input = calculator.create_empty_matrix_input(rows2, cols2, "B")
            matrix_b = calculator.parse_matrix_input(matrix_b_input, rows2, cols2)

            if matrix_b is not None:
                display_matrix(matrix_b, "Матрица B")
        elif operation == "Умножение на скаляр":
            st.header("Скаляр")
            scalar = st.number_input("Введите скаляр", value=1.0)

    # Кнопка выполнения операции
    if st.button("Выполнить операцию", type="primary"):
        if matrix_a is None:
            st.error("Пожалуйста, введите корректную матрицу A!")
            return

        result = None

        # Выполнение выбранной операции
        if operation == "Транспонирование":
            result = calculator.transpose(matrix_a)

        elif operation == "Умножение матриц":
            if matrix_b is not None:
                result = calculator.multiply_matrices(matrix_a, matrix_b)

        elif operation == "Сложение матриц":
            if matrix_b is not None:
                result = calculator.add_matrices(matrix_a, matrix_b)

        elif operation == "Вычитание матриц":
            if matrix_b is not None:
                result = calculator.subtract_matrices(matrix_a, matrix_b)

        elif operation == "Умножение на скаляр":
            result = calculator.scalar_multiply(matrix_a, scalar)

        elif operation == "Определитель":
            det = calculator.determinant(matrix_a)
            if det is not None:
                st.success(f"**Определитель матрицы A:** {det:.6f}")

        elif operation == "Обратная матрица":
            result = calculator.inverse_matrix(matrix_a)

        elif operation == "Ранг матрицы":
            rank = calculator.matrix_rank(matrix_a)
            if rank is not None:
                st.success(f"**Ранг матрицы A:** {rank}")

        elif operation == "Собственные значения":
            eigenvalues, eigenvectors = calculator.eigenvalues_eigenvectors(matrix_a)
            if eigenvalues is not None:
                st.subheader("Собственные значения:")
                st.write(eigenvalues)

                st.subheader("Собственные векторы:")
                display_matrix(eigenvectors, "Матрица собственных векторов")

        elif operation == "Решение СЛАУ":
            st.info("Матрица A - коэффициенты системы, матрица B - вектор констант")
            if matrix_b is not None:
                result = calculator.solve_linear_system(matrix_a, matrix_b)

        # Отображение результата
        if result is not None:
            display_matrix(result, "Результат")

            # Кнопка для копирования результата
            result_str = np.array2string(result, precision=6, separator='\t')
            st.code(f"Результат:\n{result_str}", language='text')


def render_history(calculator):
    """Рендер истории вычислений"""
    st.header("История вычислений")

    if not calculator.history:
        st.info("История вычислений пуста")
        return

    # Отображаем историю в обратном порядке (последние операции первыми)
    for i, item in enumerate(reversed(calculator.history)):
        with st.expander(f"Операция {len(calculator.history) - i}: {item['operation']}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                if item['matrix1'] is not None:
                    display_matrix(item['matrix1'], "Матрица 1")

            with col2:
                if item['matrix2'] is not None:
                    display_matrix(item['matrix2'], "Матрица 2")
                else:
                    st.write("—")

            with col3:
                if item['result'] is not None:
                    display_matrix(item['result'], "Результат")

            # Время выполнения
            st.caption(f"Время: {time.ctime(item['timestamp'])}")


def render_help():
    """Рендер справки"""
    st.header("📚 Справка по MatrixLab")

    st.markdown("""
    ### Как пользоваться калькулятором:

    1. **Ввод матриц**: Вводите элементы матрицы построчно, разделяя числа пробелами или запятыми
    2. **Выбор операции**: Выберите нужную операцию из выпадающего списка
    3. **Выполнение**: Нажмите кнопку "Выполнить операцию"

    ### Поддерживаемые операции:

    **Базовые операции:**
    - 🔄 Транспонирование
    - ✖️ Умножение матриц  
    - ➕ Сложение матриц
    - ➖ Вычитание матриц
    - 🔢 Умножение на скаляр

    **Аналитические операции:**
    - 📐 Определитель матрицы
    - 🔄 Обратная матрица
    - 📊 Ранг матрицы

    **Продвинутые операции:**
    - 🎯 Собственные значения и векторы
    - 🧮 Решение систем линейных уравнений

    ### Примеры ввода:
    ```
    Матрица 2×2:
    1 2
    3 4

    Матрица 3×2:
    1, 2, 3
    4, 5, 6
    ```
    """)


if __name__ == "__main__":
    main()