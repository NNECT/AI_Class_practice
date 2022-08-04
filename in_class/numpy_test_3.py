import numpy as np


if __name__ == "__main__":
    # a = np.arange(3)
    # b = np.arange(6)
    # c = np.arange(3).reshape(1, 3)
    # d = np.arange(6).reshape(2, 3)
    # e = np.arange(3).reshape(3, 1)
    #
    # print(a + e)

    np.random.seed(23)

    g = np.random.randint(0, 100, 12).reshape(-1, 4)
    print(g, end='\n\n')

    print(np.max(g))            # 최댓값
    print(np.max(g, axis=0))
    print(np.max(g, axis=1))
    print()

    print(np.argmax(g))         # 최댓값의 인덱스
    print(np.argmax(g, axis=0))
    print(np.argmax(g, axis=1))
    print()

    print(np.sum(g))            # 합계
    print(np.sum(g, axis=0))
    print(np.sum(g, axis=1))
    print()

    print(np.cumsum(g))         # 누적합
    print(np.cumsum(g, axis=0))
    print(np.cumsum(g, axis=1))
    print()

    # print(np.cumprod(g))         # 누적곱
    print(np.cumprod(g, axis=0))
    print(np.cumprod(g, axis=1))
    print()

    t1 = np.arange(12).reshape(3, 4)
    t2 = np.arange(6).reshape(3, 2)
    t3 = np.arange(8).reshape(2, 4)
    print(t1)
    print(t2)
    print(np.hstack([t1, t2]))      # 합치기 h(horizontal)
    print(np.concatenate([t1, t2], axis=1))
    print(np.vstack([t1, t3]))      # v(vertical)
    print(np.concatenate([t1, t3], axis=0))

    print(np.transpose(t1))         # 전치
