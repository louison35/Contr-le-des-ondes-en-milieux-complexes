import matplotlib.pyplot as plt
import numpy as np
import random as rd


a = 0.1
aa = 100
b = 0.4
bb = 400
c0 = 3 * 10**8

def mode(f1):
    omega = 2*np.pi*f1
    Ni = 1
    fi = 0
    A = True

    while A:
        fi = (c0 * Ni) / (2 * a)
        if fi > f1:
            A = False
        else:
            A = True
            Ni += 1

    N = Ni - 1
    k = [np.sqrt((omega / c0) ** 2 - ((i+1) * (np.pi / a)) ** 2) for i in range(N)]
    
    return k,N


def green_in(x1, y1, k1, N):
    g1 = np.zeros((N, len(x1)), dtype=complex)
    g2 = np.zeros((N, len(x1)), dtype=complex)

    for i in range(N):
        for j in range(len(x1)):
            g1[i, j] = (
                np.sqrt(2 / a)
                * ((-1) ** i)
                * (1 / np.sqrt(k1[i]))
                * np.sin(((i+1)*(np.pi/a))* y1[j])
                * np.exp(1j * k1[i] * x1[j])
            )

            g2[i, j] = (
                np.sqrt(2 / a)
                * ((-1) ** i)
                * (1 / np.sqrt(k1[i]))
                * np.sin(((i+1)*(np.pi/a)) * y1[j])
                * np.exp(1j * k1[i] * abs(b - x1[j]))
            )

    return np.concatenate((g1, g2))


def green_dd(x1, y1, k1, N):
    gi = np.zeros((len(x1), len(x1)), dtype=complex)
    g = np.zeros((len(x1), len(x1)), dtype=complex)
    for o in range(N):
        for i in range(len(x1)):
            for j in range(len(x1)):
                gi[i, j] = (
                    -np.sin(((o+1)*(np.pi/a)) * y1[i])
                    * np.sin(((o+1)*(np.pi/a)) * y1[j])
                    * (2 *(1/a) *( 1/k1[o]))
                    * np.exp(1j * k1[o] * abs(x1[j] - x1[i]))
                )
        g = g + gi
    return g


def S_matrix(x1, y1, k1, alpha, N):
    G1 = green_in(x1, y1, k1, N)
    G_t1 = np.transpose(G1)
    G_dd1 = green_dd(x1, y1, k1, N)
    w1 = alpha - G_dd1
    w_inv1 = np.linalg.inv(w1)
    S1 = np.dot(G1, np.dot(w_inv1, G_t1))
    S0 = np.zeros((2 * N, 2 * N), dtype=complex)
    G0 = [np.exp(1j * b * k1[o]) for o in range(N)]
    t0 = np.diag(G0)
    S0[:N, N : 2 * N] = t0
    S0[N : 2 * N, :N] = t0
    return S1-S0


def M_alpha(a_tot):
    alphai = np.zeros((len(a_tot), len(a_tot)), dtype=complex)
    for i in range(len(a_tot)):
        alphai[i, i] = 1 / -(a_tot[i] * 1j)
    result = alphai
    return result

def coef_np(M,N,i):
    M = abs(M)
    t = M[N:,:N]
    return t[i,i]


x =  [0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667, 0.1, 0.1, 0.1, 0.1, 0.1, 0.13333333333333333, 0.13333333333333333, 0.13333333333333333, 0.13333333333333333, 0.13333333333333333, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.298, 0.355, 0.213]
y =  [0.016666666666666666, 0.03333333333333333, 0.05, 0.06666666666666667, 0.08333333333333333, 0.016666666666666666, 0.03333333333333333, 0.05, 0.06666666666666667, 0.08333333333333333, 0.016666666666666666, 0.03333333333333333, 0.05, 0.06666666666666667, 0.08333333333333333, 0.016666666666666666, 0.03333333333333333, 0.05, 0.06666666666666667, 0.08333333333333333, 0.016666666666666666, 0.03333333333333333, 0.05, 0.06666666666666667, 0.08333333333333333, 0.076, 0.031, 0.007]
params = [0.27588826417922974, 13.494832038879395, 9.999999974752427e-07, 13.159795761108398, 9.999999974752427e-07, 9.999999974752427e-07, 0.4111560583114624, 0.3325090706348419, 9.999999974752427e-07, 9.999999974752427e-07, 0.9305669069290161, 9.999999974752427e-07, 2.255425453186035, 9.999999974752427e-07, 2.65584659576416, 1.5741199254989624, 9.999999974752427e-07, 0.5450328588485718, 9.999999974752427e-07, 3.018786907196045, 0.40817245841026306, 0.37347447872161865, 0.15430137515068054, 9.999999974752427e-07, 9.999999974752427e-07, -6.0, -6.0, -6.0]

alpha = M_alpha(params)

x0 = [i for i in x[-3:]]
y0 = [i for i in y[-3:]]
par0 = [i for i in params[-3:]]
alpha0 = M_alpha(par0)

def test(f):
    ft = np.linspace(f-0.49,f+0.49,500)
    t = []
    t0 = []
    for i in ft:
        fi = i*10**9
        k,N = mode(fi)
        Si = S_matrix(x, y, k, alpha, N)
        S0i = S_matrix(x0, y0, k, alpha0, N)
        ti = coef_np(Si, N,2)
        t0i = coef_np(S0i, N,2)
        t.append(ti)
        t0.append(t0i)
    return ft,t,t0

f,t,t0= test(7)

plt.rcParams['figure.dpi'] = 300
plt.plot(f,t,"-",label = "Optimisé")
plt.plot(f,t0,"-",label = "Non optimisé")
plt.xlabel("Fréquence du signal (GHz)")
plt.ylabel("Transmission <T>")
plt.ylim((0.0,1))
plt.grid()
plt.legend()
plt.show()




    
