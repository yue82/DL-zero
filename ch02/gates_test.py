# -*- coding: utf-8 -*-
from gates import AND, OR, NAND, XOR
from itertools import product


def gate_test(gate, xs_list):
    for xs in xs_list:
        y = gate(xs[0], xs[1])
        print(('{}({}) -> {}').format(gate.__name__, xs, y))


def main():
    xs_list = list(product([0, 1], repeat=2))
    gate_test(AND, xs_list)
    gate_test(OR, xs_list)
    gate_test(NAND, xs_list)
    gate_test(XOR, xs_list)


if __name__ == '__main__':
    main()
