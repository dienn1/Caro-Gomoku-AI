import caro
import random
import time
import timeit
import cython
import cProfile


def test(count: int = 1000, log: bool = True):
    win_count: int = 0
    tie_count: int = 0
    t = time.time()
    b_main: caro.Caro = caro.Caro()
    b: caro.Caro = caro.Caro()
    for i in range(count):
        b.copy_from(b_main)
        # b: caro.Caro = caro.Caro()
        b.simulate()
        if b.get_state() == 1:
            win_count += 1
        elif b.get_state() == 0:
            tie_count += 1
    if log:
        print(time.time() - t)
        print("X WON", win_count)
        print("O WON", count - win_count - tie_count)


if __name__ == "__main__":
    # cProfile.run("test3(20)")
    # cProfile.run("caro.test()")

    while True:
        test()

    # b1 = caro.Caro(10)
    # print("PLAY B1 15 turns")
    # b1.simulate(15)
    # print(b1)
    # b2 = caro.Caro(10)
    # print("COPY B1 TO B2")
    # b2.copy_from(b1)
    # print(b2)
    # print("SIMULATE B2")
    # b2.simulate()
    # print("B1")
    # print(b1, "\n")
    # print("B2")
    # print(b2, "\n")
    #
    # print("COPY B1 TO B2")
    # b2.copy_from(b1)
    # print(b2)
    # print("SIMULATE B2")
    # b2.simulate()
    # print("B1")
    # print(b1, "\n")
    # print("B2")
    # print(b2, "\n")

