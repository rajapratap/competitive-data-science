import math
from itertools import permutations

def header():
    return 'YANDEX EXAM';

def run():

    #x1, x2, x3, x4 = -14,10,-60,11
    #print(2*x1+0*x2+1*x3+8*x4)
    #
    #
    #return

    n = 0

    bag = set([1, 2, 3, 4, 5])

    for i1 in bag:
        used = [i1]
        for i2 in range(1, 6):
            for i3 in range(1, 6):
                for i4 in range(1, 6):
                    for i5 in range(1, 6):
                        a = [i1, i2, i3, i4, i5]
                        s = set(a)
                        if (len(a) > len(s)):
                            continue

                        b = False;
                        for i in range(4):
                            if a[i+1] == (a[i]+1):
                                b = True
                                break

                        if b:
                            continue

                        n += 1
                        print(a)

    print(n)





    for N in range(2, 20):
        tot = 0
        n = 0
        for p in permutations(range(1, N+1)):
            a = list(p)

            tot += 1
            b = False
            for i in range(N-1):
                if a[i+1] == a[i]+1:
                    b = True
                    break
            if b:
                continue

            n += 1

        print(N, tot, n)

    return


    for i1 in range(1, 6):
        for i2 in range(1, 6):
            for i3 in range(1, 6):
                for i4 in range(1, 6):
                    for i5 in range(1, 6):
                        a = [i1, i2, i3, i4, i5]
                        s = set(a)
                        if (len(a) > len(s)):
                            continue

                        b = False;
                        for i in range(4):
                            if a[i+1] == (a[i]+1):
                                b = True
                                break

                        if b:
                            continue

                        n += 1
                        print(a)

    print(n)



    # SUCCESS
    n = int(input())
    a = [int(ai) for ai in input().split()]

    result = 0
    a.sort()
    i = 0

    while n>i+2:
        x = a[i] + a[n-2]
        result += x
        last = a[n-1]
        if x>=last:
            a[n-2] = a[n-1]
            a[n-1] = x
        else:
            a[n-2] = x
        i += 1

    if n>1:
        print(a[n-2] + a[n-1] + result)
    elif n==1:
        print(a[0] + result)
    else:
        print(0)

    return









    p = 2*math.pow(4*math.pi, 7)/(8*9*10*11*12*13*14*15)*(1 - 1/math.pow(2, 15))
    print(p)

    p = 1 - 1/math.pow(2, 15)

    print(math.log(0.6)/math.log(p))

    for n in range(16730, 16750):
        print(n, math.pow(p, n))

    return





    for x in [-1, 1, -2, 2, -3, 3, -4, 4, -6, 6, -12, 12, -18, 18, -9, 9]:
        print('-----')
        a = (-18 - x*x*x)/(x*x)
        b = (-12 - x*x*x)/(x)
        print(a, b)

        for r in [-1, 1, -2, 2, -3, 3, -4, 4, -6, 6, -12, 12, -18, 18, -9, 9]:
            p1 = r*r*r + a*r*r + 18
            p2 = r*r*r + b*r + 12
            if (abs(p1) < 0.001) and (abs(p2) < 0.001) and (abs(x - r) > 0.001):
              print(x, r)



    return