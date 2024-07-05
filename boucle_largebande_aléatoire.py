import torch
import torch.optim as optim
import numpy as np
import random as rd
import matplotlib.pyplot as plt


nx = 5
ny = 5
n = nx*ny
n_diff = 3
a = 0.1
aa = 100
b = 0.4
bb = 400
c0 = 3 * 10**8

Ni = 1
fi = 0
A = True

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



def green_in(x1, y1, k1,N1):
    g1 = np.zeros((N1, len(x1)), dtype=complex)
    g2 = np.zeros((N1, len(x1)), dtype=complex)

    for i in range(N1):
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


def green_dd(x1, y1, k1, N1):
    gi = np.zeros((len(x1), len(x1)), dtype=complex)
    g = np.zeros((len(x1), len(x1)), dtype=complex)
    for o in range(N1):
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


def S_matrix(x1, y1, alpha, k1,N1):
    G1 = torch.tensor(green_in(x1, y1, k1, N1), dtype=torch.cfloat)
    Gdd1 = green_dd(x1, y1, k1, N1)
    G_t1 = torch.transpose(G1, 0, 1)
    w1 = alpha - torch.tensor(Gdd1, dtype=torch.cfloat)
    w_inv1 = torch.linalg.inv(w1)
    S1 = torch.matmul(G1, torch.matmul(w_inv1, G_t1))
    S0 = np.zeros((2 * N1, 2 * N1), dtype=complex)
    G0 = [np.exp(1j * b * k1[o]) for o in range(N1)]
    t0 = np.diag(G0)
    S0[:N1, N1 : 2 * N1] = t0
    S0[N1 : 2 * N1, :N1] = t0
    S0 = torch.tensor(S0, dtype=torch.cfloat)
    return S1 - S0

def matrix(par,b1):
    ft = [6.9*10**9+i*0.1*10**9 for i in range(b1)]
    somme = []
    par = torch.cat((par,par_diff))
    par1 = 1/-(par*1j)
    alpha = torch.diag(par1)
    for i in ft:
        kj,Nj = mode(i)
        Sj = S_matrix(x_tot,y_tot,alpha,kj,Nj)
        tj = abs(Sj[Nj:,:Nj])
        valuej = (1-tj[2,2])
        somme.append(valuej)                                                              
    return sum(somme)

def matrice(par):
    par = torch.cat((par,par_diff))
    par1 = 1/-(par*1j)
    alpha = torch.diag(par1)
    return alpha



x = []
y = []

for i in range(nx):
    for j in range(ny):
        x.append(((i+1)*200/(nx+1))/1000)
        y.append(((j+1)*100/(ny+1))/1000)





par_diff = torch.tensor([-6 for i in range(n_diff)])

def coef(M,N):
    M = abs(M)
    t = M[N:,:N]
    return t[2,2]

f_test = 7*10**9
k_test,N_test = mode(f_test)
T = 0.1
while T <0.75:
    x_diff = [rd.randrange(201,400) / 1000 for i in range(n_diff)]
    y_diff = [rd.randrange(0, aa) / 1000 for i in range(n_diff)]
    S = S_matrix(x_diff, y_diff, matrice(torch.tensor([])), k_test, N_test)
    T = coef(S,N_test)
    

x_tot = x + x_diff
y_tot = y + y_diff



def coef_np(M,N):
    M = abs(M)
    t = M[N:,:N]
    t = np.dot(t,np.conj(np.transpose(t)))
    r = M[:N,:N]
    r = np.dot(r,np.conj(np.transpose(r)))
    t_moy = 0
    r_moy = 0
    for i in range(N):
        t_moy += t[i,i]
        r_moy += r[i,i]          
    return (t_moy/N),(r_moy/N)
    

num_iterations =2001       
t_moy = []
largeur = []

for p in range(3):
    params = torch.tensor([0.001*torch.rand(1).item() for _ in range(len(x))], requires_grad=True)
    optimizer = optim.Adam([params], lr=0.01)
    for iteration in range(num_iterations):
        optimizer.zero_grad()
    
        cost = matrix(params,(p+1)) 
        cost.backward() 
    
        optimizer.step() 

        with torch.no_grad():
            params.clamp_(1e-6,30)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}: <R> ={(cost).item()/(p+1):.4f}")
            
    t_moy.append(1-(cost.item()/(p+1)))
    largeur.append((p+1)*0.05*10**9)
    print(f"Final parameters: {params}")

print("x = ",x_tot)
print("y = ",y_tot)
print("params =",torch.cat((params,par_diff)).tolist())

plt.rcParams['figure.dpi'] = 300
plt.plot(largeur,t_moy)
plt.ylabel("Transmission moyenne du plateau <T>")
plt.xlabel("Largeur du plateau (Hz)")
plt.grid()
plt.show()
