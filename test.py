import numpy as np
from sympy import symbols, cos
from sympy.parsing.sympy_parser import parse_expr
from collections import namedtuple
import math

Pair = namedtuple('Pair', ['left', 'right'])

class PolyApproximation:
    function = None

    def __init__(self, dataset, power):
        self.dataset = dataset
        self.approximation, self.coef = self.get_poly_approx(power)

    @staticmethod
    def main():
        dataset = []
        a = -math.pi/2 + 0.01
        b = math.pi/2 - 0.01
        x = symbols('x')
        function_str =  "1/(cos(x))"
        PolyApproximation.function = parse_expr(function_str)
        h = (b - a) / 10
        for i in range(11):
            dataset.append(Pair(a + h * i, float(PolyApproximation.function.subs(x, a + h * i))))
        power1 = PolyApproximation(dataset, 1)
        power2 = PolyApproximation(dataset, 2)
        power5 = PolyApproximation(dataset, 5)
        power6 = PolyApproximation(dataset, 6)
        print("Таблица узлов интерполирования")
        PolyApproximation.print_data(dataset, len(dataset))
        print()
        print("Таблица коэффициентов м.н.с.п.")
        PolyApproximation.print_array(power1.coef, "n = 1")
        PolyApproximation.print_array(power2.coef, "n = 2")
        PolyApproximation.print_array(power5.coef, "n = 5")
        PolyApproximation.print_array(power6.coef, "n = 6")
        print()
        print("Таблица значений функции и м.н.с.п.")
        PolyApproximation.print_result(a, b, [power1, power2, power5, power6])

    @staticmethod
    def print_result(a, b, approximation):
        dataset = []
        h = (b - a) / 50
        x = symbols('x')
        for i in range(51):
            dataset.append(Pair(a + h * i, float(PolyApproximation.function.subs(x, a + h * i))))
        print("x y(x) P1(x) P2(x) P5(x) P6(x)")
        for dot in dataset:
            l1 = approximation[0].approximation.subs(x, dot.left)
            l2 = approximation[1].approximation.subs(x, dot.left)
            l5 = approximation[2].approximation.subs(x, dot.left)
            l6 = approximation[3].approximation.subs(x, dot.left)
            print(f"{dot.left:.2f}".replace('.', ',') + " " +
                  f"{dot.right:.2f}".replace('.', ',') + " " +
                  f"{l1:.2f}".replace('.', ',') + " " +
                  f"{l2:.2f}".replace('.', ',') + " " +
                  f"{l5:.2f}".replace('.', ',') + " " +
                  f"{l6:.2f}".replace('.', ','))

    @staticmethod
    def print_array(data, index):
        result = f"{index} "
        for number in data:
            result += f"{number:.2f}".replace('.', ',') + " "
        print(result)

    @staticmethod
    def print_data(data, length):
        x = [pair.left for pair in data]
        y = [pair.right for pair in data]
        output = "x " + " ".join([f"{x_value:.2f}".replace('.', ',') for x_value in x])
        print(output)
        output = "y " + " ".join([f"{y_value:.2f}".replace('.', ',') for y_value in y])
        print(output)

    def get_poly_approx(self, power):
        sums = np.zeros(2 * power + 1)
        cross_sums = np.zeros(power + 1)
        x_values = [pair.left for pair in self.dataset]
        for i in range(2 * power + 1):
            sums[i] = self.get_sum_of_power(x_values, i)
        for i in range(power + 1):
            cross_sums[i] = self.get_cross_power_sum(self.dataset, i)
        a = self.get_solvable_matrix(sums, cross_sums)
        b = np.array(cross_sums).reshape(-1, 1)
        coef_matrix = np.linalg.inv(a).dot(b)
        coef = coef_matrix.transpose()[0]

        expression = ""
        for i in range(power + 1):
            current = f"{coef[i]} * x**{i} + "
            expression += current
        return parse_expr(expression[:-3]), coef

    @staticmethod
    def get_sum_of_power(numbers, power):
        return sum([num ** power for num in numbers])

    @staticmethod
    def get_cross_power_sum(dataset, power):
        return sum([pair.left ** power * pair.right for pair in dataset])

    @staticmethod
    def get_solvable_matrix(sums, cross_sums):
        matrix = np.zeros((len(cross_sums), len(cross_sums)))
        for i in range(len(matrix)):
            matrix[i] = sums[i:i+len(cross_sums)]
        return matrix

if __name__ == "__main__":
    PolyApproximation.main()
