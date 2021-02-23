import csv
import numpy as np


########################
###### Questão 1 #######
########################

def plu(A, B):

    # Transforma os elementos em arrays
    B = np.transpose(B)[0]
    a, b = np.copy(A), np.copy(B)
    coluna = 0
    I = np.identity(len(A))
    # t vai ser a matriz responsavel por zerar os elementos abaixo da diagonal principal de a
    t = np.copy(I)
    while coluna < len(A[0]):
        linha = coluna + 1
        a = np.dot(t, a)
        b = np.dot(t, b)
        t = np.copy(I)
        while linha < len(A):
            # Se o pivo for = 0, o denominador zeraria, então usamos a matriz de permutação
            if a[coluna][coluna] == 0:
                a, b = permuta(a, b, coluna)
            numerador = a[linha][coluna]
            denominador = a[coluna][coluna]
            t[linha][coluna] = numerador / denominador * -1
            linha += 1
        coluna += 1
    X = calcula_valores(a, b)
    return X


def calcula_valores(a, b):
    valores = [0] * len(a[0])
    i = len(a[0]) -1
    while i >= 0:
        # Faz back substitution
        valores[i] = (b[i] - sum([a[i][j] * valores[j] for j in range(len(valores))]))/a[i][i]
        i -= 1
    resultado = [] * len(valores)
    for posicao in range(len(valores)):
        resultado.append([valores[posicao]])

    return np.array(resultado)


def permuta(a,b, linha_pivo):
    for linha in range(len(a)):
        # Só é pra trocar se a linha estiver abaixo do pivo
        if linha > linha_pivo:
            # a[linha][linha_pivo] foi usado pq a linha do pivo tem q ser igual a coluna
            if a[linha][linha_pivo] != 0:
                aux1 = np.copy(a[linha])
                a[linha] = a[linha_pivo][:]
                a[linha_pivo] = aux1[:]

                aux2 = b[linha]
                b[linha] = b[linha_pivo]
                b[linha_pivo] = aux2
                break
    return a, b


with open("dados.csv") as arq:
    csv_reader = csv.reader(arq, delimiter=',')
    iris = list(csv_reader)

setosa_a_sem = []
versicolor_a_sem = []
virginica_a_sem = []

setosa_a_com = []
versicolor_a_com = []
virginica_a_com = []

setosa_y = []
versicolor_y = []
virginica_y = []


for linha in iris:
    if linha[-1] == 'Iris-setosa':
        linha_float = [float(elemento) for elemento in linha[:-1]]
        setosa_a_sem.append(linha_float[1:-1])
        setosa_a_com.append(linha_float[1:-1]+[1])
        setosa_y.append([float(linha[-2])])


    elif linha[-1] == 'Iris-versicolor':
        linha_float = [float(elemento) for elemento in linha[:-1]]
        versicolor_a_sem.append(linha_float[1:-1])
        versicolor_a_com.append(linha_float[1:-1]+[1])
        versicolor_y.append([float(linha[-2])])
    elif linha[-1] == 'Iris-virginica':
        linha_float = [float(elemento) for elemento in linha[:-1]]
        virginica_a_sem.append(linha_float[1:-1])
        virginica_a_com.append(linha_float[1:-1]+[1])
        virginica_y.append([float(linha[-2])])


# A (sem termo independente)
setosa_a_sem, versicolor_a_sem, virginica_a_sem = np.array(setosa_a_sem), np.array(versicolor_a_sem), np.array(virginica_a_sem)


# A (com termo independente)
setosa_a_com, versicolor_a_com, virginica_a_com = np.array(setosa_a_com), np.array(versicolor_a_com), np.array(virginica_a_com)

# Y
setosa_y, versicolor_y, virginica_y = np.array(setosa_y), np.array(versicolor_y), np.array(virginica_y)

# Transpostas
setosa_a_sem_transposta, versicolor_a_sem_transposta, virginica_a_sem_transposta = np.transpose(setosa_a_sem), np.transpose(versicolor_a_sem), np.transpose(virginica_a_sem)

setosa_a_com_transposta, versicolor_a_com_transposta, virginica_a_com_transposta = np.transpose(setosa_a_com), np.transpose(versicolor_a_com), np.transpose(virginica_a_com)



#Sem termo independente
setosa_sem_at_a = np.dot(setosa_a_sem_transposta, setosa_a_sem)
setosa_sem_at_y = np.dot(setosa_a_sem_transposta, setosa_y)

versicolor_sem_at_a = np.dot(versicolor_a_sem_transposta, versicolor_a_sem)
versicolor_sem_at_y = np.dot(versicolor_a_sem_transposta, versicolor_y)

virginica_sem_at_a = np.dot(virginica_a_sem_transposta, virginica_a_sem)
virginica_sem_at_y = np.dot(virginica_a_sem_transposta, virginica_y)

# Com termo independente
setosa_com_at_a = np.dot(setosa_a_com_transposta, setosa_a_com)
setosa_com_at_y = np.dot(setosa_a_com_transposta, setosa_y)

versicolor_com_at_a = np.dot(versicolor_a_com_transposta, versicolor_a_com)
versicolor_com_at_y = np.dot(versicolor_a_com_transposta, versicolor_y)

virginica_com_at_a = np.dot(virginica_a_com_transposta, virginica_a_com)
virginica_com_at_y = np.dot(virginica_a_com_transposta, virginica_y)


# PLU

# Sem
coeficiente_setosa_sem = plu(setosa_sem_at_a, setosa_sem_at_y)
coeficiente_versicolor_sem = plu(versicolor_sem_at_a, versicolor_sem_at_y)
coeficiente_virginica_sem = plu(virginica_sem_at_a, virginica_sem_at_y)

# Com
coeficiente_setosa_com = plu(setosa_com_at_a, setosa_com_at_y)
coeficiente_versicolor_com = plu(versicolor_com_at_a, versicolor_com_at_y)
coeficiente_virginica_com = plu(virginica_com_at_a, virginica_com_at_y)

"""
print("Setosa - Sem")
print(coeficiente_setosa_sem)
print("\nVersicolor - Sem")
print(coeficiente_versicolor_sem)
print("\nVirginica - Sem")
print(coeficiente_virginica_sem)
print("\nSetosa - Com")
print(coeficiente_setosa_com)
print("\nVersicolor - Com")
print(coeficiente_versicolor_com)
print("\nVirginica - Com")
print(coeficiente_virginica_com)
"""
coeficientes_sem = {
    'setosa_sem': coeficiente_setosa_sem,
    'versicolor_sem': coeficiente_versicolor_sem,
    'virginica_sem': coeficiente_virginica_sem,

}

coeficientes_com = {
    'setosa_com': coeficiente_setosa_com,
    'versicolor_com': coeficiente_versicolor_com,
    'virginica_com': coeficiente_virginica_com
}


########################
###### Questão 2 #######
########################

# Calculando autovalores e autovetores
autovalores_setosa_com, autovetores_setosa_com = np.linalg.eig(setosa_com_at_a)
autovalores_setosa_sem, autovetores_setosa_sem = np.linalg.eig(setosa_sem_at_a)
autovalores_versicolor_com, autovetores_versicolor_com = np.linalg.eig(versicolor_com_at_a)
autovalores_versicolor_sem, autovetores_versicolor_sem = np.linalg.eig(versicolor_sem_at_a)
autovalores_virginica_com, autovetores_virginica_com = np.linalg.eig(virginica_com_at_a)
autovalores_virginica_sem, autovetores_virginica_sem = np.linalg.eig(virginica_sem_at_a)

# Diagonalizando os autovalores
lambda_setosa_com = np.diag(autovalores_setosa_com)
lambda_setosa_sem = np.diag(autovalores_setosa_sem)
lambda_versicolor_com = np.diag(autovalores_versicolor_com)
lambda_versicolor_sem = np.diag(autovalores_versicolor_sem)
lambda_virginica_com = np.diag(autovalores_virginica_com)
lambda_virginica_sem = np.diag(autovalores_virginica_sem)

# Decomposição espectral

espectral_setosa_com = f"V -\n{autovetores_setosa_com},\n\nΛ -\n{lambda_setosa_com},\n\nV^t -\n{np.transpose(autovetores_setosa_com)}"
espectral_setosa_sem = f"V -\n{autovetores_setosa_sem},\n\nΛ -\n{lambda_setosa_sem},\n\nV^t -\n{np.transpose(autovetores_setosa_sem)}"

espectral_versicolor_com = f"V -\n{autovetores_versicolor_com},\n\nΛ -\n{lambda_versicolor_com},\n\nV^t -\n{np.transpose(autovetores_versicolor_com)}"
espectral_versicolor_sem = f"V -\n{autovetores_versicolor_sem},\n\nΛ -\n{lambda_versicolor_sem},\n\nV^t -\n{np.transpose(autovetores_versicolor_sem)}"

espectral_virginica_com = f"V -\n{autovetores_virginica_com},\n\nΛ -\n{lambda_virginica_com},\n\nV^t -\n{np.transpose(autovetores_virginica_com)}"
espectral_virginica_sem = f"V -\n{autovetores_virginica_sem},\n\nΛ -\n{lambda_virginica_sem},\n\nV^t -\n{np.transpose(autovetores_virginica_sem)}"

#Só pra testar
print(setosa_com_at_a)
print(np.dot(autovetores_setosa_com,lambda_setosa_com).dot(np.transpose(autovetores_setosa_com)))




print("Setosa - com termo independente")
print(espectral_setosa_com)
print('--------------------------------------------------')
print("\n\nSetosa - sem termo independente")
print(espectral_setosa_sem)
print('--------------------------------------------------')
print("\n\nVersicolor - com termo independente")
print(espectral_versicolor_com)
print('--------------------------------------------------')
print("\n\nVersicolor - sem termo independente")
print(espectral_versicolor_sem)
print('--------------------------------------------------')
print("\n\nVirginica - com termo independente")
print(espectral_virginica_com)
print('--------------------------------------------------')
print("\n\nVirginica - sem termo independente")
print(espectral_virginica_sem)
print('----------------------------------------')

########################
###### Questão 3 #######
########################

(U_setosa_com, s_setosa_com, Vt_setosa_com) = np.linalg.svd(setosa_com_at_a)
svd_setosa_com = f"U -\n{U_setosa_com},\n\n Σ -\n{np.diag(s_setosa_com)}\n\nV^t-\n{Vt_setosa_com}"
(U_setosa_sem, s_setosa_sem, Vt_setosa_sem) = np.linalg.svd(setosa_sem_at_a)
svd_setosa_sem = f"U -\n{U_setosa_sem},\n\n Σ -\n{np.diag(s_setosa_sem)}\n\nV^t-\n{Vt_setosa_sem}"

(U_versicolor_com, s_versicolor_com, Vt_versicolor_com) = np.linalg.svd(versicolor_com_at_a)
svd_versicolor_com = f"U -\n{U_versicolor_com},\n\n Σ -\n{np.diag(s_versicolor_com)}\n\nV^t-\n{Vt_versicolor_com}"
(U_versicolor_sem, s_versicolor_sem, Vt_versicolor_sem) = np.linalg.svd(versicolor_sem_at_a)
svd_versicolor_sem = f"U -\n{U_versicolor_sem},\n\n Σ -\n{np.diag(s_versicolor_sem)}\n\nV^t-\n{Vt_versicolor_sem}"


(U_virginica_com, s_virginica_com, Vt_virginica_com) = np.linalg.svd(virginica_com_at_a)
svd_virginica_com = f"U -\n{U_virginica_com},\n\n Σ -\n{np.diag(s_virginica_com)}\n\nV^t-\n{Vt_virginica_com}"
(U_virginica_sem, s_virginica_sem, Vt_virginica_sem) = np.linalg.svd(virginica_sem_at_a)
svd_virginica_sem = f"U -\n{U_virginica_sem},\n\n Σ -\n{np.diag(s_virginica_sem)}\n\nV^t-\n{Vt_virginica_sem}"
print(np.dot)

"""
print("Setosa - com termo independente")
print(svd_setosa_com)
print('--------------------------------------------------')
print("\n\nSetosa - sem termo independente")
print(svd_setosa_sem)
print('--------------------------------------------------')
print("\n\nVersicolor - com termo independente")
print(svd_versicolor_com)
print('--------------------------------------------------')
print("\n\nVersicolor - sem termo independente")
print(svd_versicolor_sem)
print('--------------------------------------------------')
print("\n\nVirginica - com termo independente")
print(svd_virginica_com)
print('--------------------------------------------------')
print("\n\nVirginica - sem termo independente")
print(svd_virginica_sem)
print('--------------------------------------------------')
"""



"""
print(np.allclose(np.dot(U_setosa_com,np.diag(s_setosa_com)).dot(Vt_setosa_com), setosa_com_xt_x))
print(np.allclose(np.dot(U_setosa_sem,np.diag(s_setosa_sem)).dot(Vt_setosa_sem), setosa_sem_xt_x))

print(np.allclose(np.dot(U_versicolor_com,np.diag(s_versicolor_com)).dot(Vt_versicolor_com), versicolor_com_xt_x))
print(np.allclose(np.dot(U_versicolor_sem,np.diag(s_versicolor_sem)).dot(Vt_versicolor_sem), versicolor_sem_xt_x))

print(np.allclose(np.dot(U_virginica_com,np.diag(s_virginica_com)).dot(Vt_virginica_com), virginica_com_xt_x))
print(np.allclose(np.dot(U_virginica_sem,np.diag(s_virginica_sem)).dot(Vt_virginica_sem), virginica_sem_xt_x))
"""
########################
###### Questão 4 #######
########################
A = [5.0, 2.3, 3.3, 1.0]
B = [4.6, 3.2, 1.4, 0.2]
C = [5.0, 3.3, 1.4, 0.2]
D = [6.1, 3.0, 4.6, 1.4]
E = [5.9, 3.0, 5.1, 1.8]
F = [7.6, 2.9, 6.8, 2.3]


def calcula_erro(amostra, coeficientes):
    erros = {}
    for coeficiente in coeficientes:
        indice = 0
        soma = 0
        while indice < len(coeficientes[coeficiente]):
            if indice == 3:
                soma += coeficientes[coeficiente][indice][0]
            else:
                soma += coeficientes[coeficiente][indice][0] * amostra[indice]
            indice += 1
        erros[coeficiente] = abs(amostra[-1] - soma)
    return erros

erroA_com = calcula_erro(A, coeficientes_com)
erroB_com = calcula_erro(B, coeficientes_com)
erroC_com = calcula_erro(C, coeficientes_com)
erroD_com = calcula_erro(D, coeficientes_com)
erroE_com = calcula_erro(E, coeficientes_com)


erroA_sem = calcula_erro(A, coeficientes_sem)
erroB_sem = calcula_erro(B, coeficientes_sem)
erroC_sem = calcula_erro(C, coeficientes_sem)
erroD_sem = calcula_erro(D, coeficientes_sem)
erroE_sem = calcula_erro(E, coeficientes_sem)

def mostra_especie(erros, termo='com'):
    especie = f'setosa_{termo}'
    menor = erros[especie]
    for erro in erros:
        if erros[erro] < menor:
            especie = erro
            menor = erros[erro]
    print(f"{especie[:-4].title()}")


"""
print("Com termo independente")
print()
print('A', end=' - ')
mostra_especie(erroA_com)
print()
print('B', end=' - ')
mostra_especie(erroB_com)

print()
print('C', end=' - ')
mostra_especie(erroC_com)

print()
print('D', end=' - ')
mostra_especie(erroD_com)

print()
print('E', end=' - ')
mostra_especie(erroE_com)

print()
print("Sem termo independente")
print()
print('A', end=' - ')
mostra_especie(erroA_sem,'sem')
print()
print('B', end=' - ')
mostra_especie(erroB_sem,'sem')

print()
print('C', end=' - ')
mostra_especie(erroC_sem,'sem')

print()
print('D', end=' - ')
mostra_especie(erroD_sem,'sem')

print()
print('E', end=' - ')
mostra_especie(erroE_sem,'sem')
"""