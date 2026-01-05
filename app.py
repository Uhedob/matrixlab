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
    st.set_page_config(page_title="Справка по матричным операциям", page_icon="📊")
    st.title("📊 Справка по матричным операциям")
    st.markdown("---")

    # Базовые операции
    st.header("🔧 Базовые операции")

    # Транспонирование
    st.subheader("🔄 Транспонирование")
    st.write("**Определение:** Меняет строки и столбцы матрицы местами. Элемент Aᵀ[i][j] = A[j][i]")
    st.write("**Пример:**")
    st.code("""
    Дана матрица A:
    [1  2  3]
    [4  5  6]
    
    Транспонированная матрица Aᵀ:
    [1  4]
    [2  5]
    [3  6]
    
    Пошагово:
    1. Первая строка [1, 2, 3] становится первым столбцом [1, 4]
    2. Вторая строка [4, 5, 6] становится вторым столбцом [2, 5]
    3. Третьего столбца нет, поэтому матрица становится 3×2
    """)

    st.markdown("---")

    # Умножение матриц
    st.subheader("✖️ Умножение матриц")
    st.write("**Определение:** Умножение строк первой матрицы на столбцы второй. C[i][j] = Σ(A[i][k] × B[k][j])")
    st.write("**Пример:**")
    st.code("""
    Даны матрицы:
    A = [1  2]    B = [5  6]
        [3  4]        [7  8]
    
    A × B = ?
    1. Элемент C[1][1] = (1×5) + (2×7) = 5 + 14 = 19
    2. Элемент C[1][2] = (1×6) + (2×8) = 6 + 16 = 22
    3. Элемент C[2][1] = (3×5) + (4×7) = 15 + 28 = 43
    4. Элемент C[2][2] = (3×6) + (4×8) = 18 + 32 = 50
    
    Результат:
    [19  22]
    [43  50]
    """)

    st.markdown("---")

    # Сложение матриц
    st.subheader("➕ Сложение матриц")
    st.write("**Опреденение:** Сложение соответствующих элементов матриц одинакового размера. C[i][j] = A[i][j] + B[i][j]")
    st.write("**Пример:**")
    st.code("""
    Даны матрицы:
    A = [1  2]    B = [5  6]
        [3  4]        [7  8]
    
    A + B = ?
    1. Элемент C[1][1] = 1 + 5 = 6
    2. Элемент C[1][2] = 2 + 6 = 8
    3. Элемент C[2][1] = 3 + 7 = 10
    4. Элемент C[2][2] = 4 + 8 = 12
    
    Результат:
    [6   8]
    [10  12]
    """)

    st.markdown("---")

    # Вычитание матриц
    st.subheader("➖ Вычитание матриц")
    st.write("**Определение:** Вычитание соответствующих элементов матриц одинакового размера. C[i][j] = A[i][j] - B[i][j]")
    st.write("**Пример:**")
    st.code("""
    Даны матрицы:
    A = [1  2]    B = [5  6]
        [3  4]        [7  8]
    
    A - B = ?
    1. Элемент C[1][1] = 1 - 5 = -4
    2. Элемент C[1][2] = 2 - 6 = -4
    3. Элемент C[2][1] = 3 - 7 = -4
    4. Элемент C[2][2] = 4 - 8 = -4
    
    Результат:
    [-4  -4]
    [-4  -4]
    """)

    st.markdown("---")

    # Умножение на скаляр
    st.subheader("🔢 Умножение на скаляр")
    st.write("**Определение:** Умножение каждого элемента матрицы на число. B[i][j] = k × A[i][j]")
    st.write("**Пример:**")
    st.code("""
    Дана матрица A и скаляр k = 3:
    A = [1  2]
        [3  4]
    
    3 × A = ?
    1. Элемент B[1][1] = 3 × 1 = 3
    2. Элемент B[1][2] = 3 × 2 = 6
    3. Элемент B[2][1] = 3 × 3 = 9
    4. Элемент B[2][2] = 3 × 4 = 12
    
    Результат:
    [3   6]
    [9  12]
    """)

    st.markdown("---")

    # Аналитические операции
    st.header("📊 Аналитические операции")

    # Определитель
    st.subheader("📐 Определитель матрицы")
    st.write("**Определение:** Скалярная величина, характеризующая квадратную матрицу. Для 2×2: det(A) = a×d - b×c")
    st.write("**Пример:**")
    st.code("""
    Дана матрица 2×2:
    A = [1  2]
        [3  4]
    
    det(A) = ?
    det(A) = (1 × 4) - (2 × 3)
           = 4 - 6
           = -2
    
    Для матрицы 3×3 используется правило Саррюса или разложение по строке/столбцу.
    """)

    st.markdown("---")

    # Обратная матрица
    st.subheader("🔄 Обратная матрица")
    st.write("**Определение:** Матрица A⁻¹, такая что A × A⁻¹ = I, где I - единичная матрица. Существует только если det(A) ≠ 0")
    st.write("**Пример:**")
    st.code("""
    Дана матрица 2×2:
    A = [1  2]
        [3  4]
    
    1. Находим определитель: det(A) = (1×4) - (2×3) = -2
    2. Меняем местами a и d: [4  2]
    3. Меняем знаки b и c: [4  -2]
                           [-3  1]
    4. Делим на определитель: A⁻¹ = (1/-2) × [4  -2] = [-2   1]
                                             [-3   1]   [1.5 -0.5]
    
    Проверка: A × A⁻¹ = [1  2] × [-2   1] = [1  0] = I
                        [3  4]   [1.5 -0.5]  [0  1]
    """)

    st.markdown("---")

    # Ранг матрицы
    st.subheader("📊 Ранг матрицы")
    st.write("**Определение:** Максимальное количество линейно независимых строк или столбцов")
    st.write("**Пример:**")
    st.code("""
    Дана матрица:
    A = [1  2  3]
        [2  4  6]
        [1  0  1]
    
    Находим ранг:
    1. Приводим к ступенчатому виду:
       [1  2  3]
       [0  0  0]  (вторая строка = 2 × первая строка)
       [0 -2 -2]  (третья - первая)
    
    2. Меняем местами строки 2 и 3:
       [1   2   3]
       [0  -2  -2]
       [0   0   0]
    
    3. Ненулевых строк: 2
       Ранг(A) = 2
    """)

    st.markdown("---")

    # Продвинутые операции
    st.header("🚀 Продвинутые операции")

    # Собственные значения и векторы
    st.subheader("🎯 Собственные значения и векторы")
    st.write("**Определение:** Числа λ и векторы v, для которых A·v = λ·v. Находятся из уравнения det(A - λI) = 0")
    st.write("**Пример:**")
    st.code("""
    Дана матрица:
    A = [2  1]
        [1  2]
    
    1. Решаем: det(A - λI) = 0
       |2-λ  1 | = 0
       |1   2-λ|
    
    2. (2-λ)² - 1 = 0
       λ² - 4λ + 3 = 0
    
    3. Корни: λ₁ = 1, λ₂ = 3
    
    4. Для λ₁ = 1:
       (A - I)v₁ = 0
       [1  1][x] = [0]
       [1  1][y]   [0]
       v₁ = [1, -1]ᵀ
    
    5. Для λ₂ = 3:
       (A - 3I)v₂ = 0
       [-1  1][x] = [0]
       [ 1 -1][y]   [0]
       v₂ = [1, 1]ᵀ
    """)

    st.markdown("---")

    # Решение систем линейных уравнений
    st.subheader("🧮 Решение систем линейных уравнений")
    st.write("**Определение:** Нахождение вектора x в системе уравнений A·x = b")
    st.write("**Пример:**")
    st.code("""
    Решаем систему:
    2x + y = 5
    x - 3y = -5
    
    1. В матричной форме: A·x = b
       A = [2   1]   x = [x]   b = [5]
           [1  -3]       [y]       [-5]
    
    2. Методом Гаусса:
       [2   1 | 5]
       [1  -3 |-5]
    
    3. Меняем строки местами:
       [1  -3 |-5]
       [2   1 | 5]
    
    4. Вычитаем 2×первую строку из второй:
       [1  -3 |-5]
       [0   7 |15]
    
    5. Из второй строки: 7y = 15 → y = 15/7 ≈ 2.14
    6. Из первой строки: x - 3y = -5 → x = -5 + 3y = -5 + 45/7 = 10/7 ≈ 1.43
    
    Решение: x ≈ 1.43, y ≈ 2.14
    """)

    st.markdown("---")
    st.info("💡 **Примечание:** Все примеры приведены для наглядности. Реальные вычисления могут использовать более сложные алгоритмы.")


if __name__ == "__main__":
    main()
