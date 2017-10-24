from sklearn.neural_network import MLPClassifier
import pandas
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def leer_database():
    """
    Encargada de realizar la lectura de los archivos csv, uniendo de este 
    modo los indivuos perteneceientes a todos los tramos de fonasa con los
    que no pertencen, procediendo a realizar su preprocesamiento para poder
    con las variables categoricas
    :return: un dataframe con todos los datos preprocesados
    """
    data = pandas.read_csv('./NO_C_2017.csv')
    data1 = pandas.read_csv('./A.csv')
    data2 = pandas.read_csv('./B.csv')
    data3 = pandas.read_csv('./C.csv')
    data4 = pandas.read_csv('./D.csv')

    # unir
    datos = pandas.concat([data, data1, data2, data3, data4])

    # mesclar los datos
    datos = datos.sample(frac=1)
    datos = datos.sample(frac=1)
    datos = datos.sample(frac=1)

    datos = datos.reset_index(drop=True)

    selectOpt = ['status_salud_publica',
                 'ingreso_mensual',
                 'edad',
                 'comuna',
                 'estrato',
                 'avaluo_auto',
                 'avaluo_bbrr',
                 'cant_personas_fam',
                 'cant_hijos_fam',
                 'tot_mont',
                 'clean2',
                 'n_rubros',
                 'tot_docs',
                 'est_civil']

    datos.loc[((datos["est_civil"] == " ") & (
    datos["cant_hijos_fam"] == 0)), "est_civil"] = "SOLTERO"
    datos.loc[((datos["est_civil"] == " ") & (
    datos["cant_hijos_fam"] > 0)), "est_civil"] = "CASADO"
    # son pasados los estados a una variable binaria
    datos.loc[datos["status_salud_publica"] == "S", "status_salud_publica"] = 1
    datos.loc[datos["status_salud_publica"] == "N", "status_salud_publica"] = 0

    # cambair sexo de la persona
    datos.loc[datos["sexo_desc"] == "F", "sexo_desc"] = 0
    datos.loc[datos["sexo_desc"] == "M", "sexo_desc"] = 1
    datos.loc[datos["sexo_desc"] == "SI", "sexo_desc"] = 0

    # ind_interd
    datos.loc[datos["ind_interd"] == " ", "ind_interd"] = 0
    datos.loc[datos["ind_interd"] == "N", "ind_interd"] = 0
    datos.loc[datos["ind_interd"] == "S", "ind_interd"] = 1

    # gente de fuera de santiago agregar comuna
    datos.loc[datos["est_civil"] == "SOLTERO", "est_civil"] = 0
    datos.loc[datos["est_civil"] == "CASADO", "est_civil"] = 3
    datos.loc[datos["est_civil"] == "VIUDO", "est_civil"] = 1
    datos.loc[datos["est_civil"] == "DIVORCIADO", "est_civil"] = 2
    datos.loc[datos["est_civil"] == "SEPARADO JUDICIALMENTE", "est_civil"] = 2

    # son discretizados los estratos sociales
    datos.loc[datos["estrato"] == "SIN CLASIFICACION", "estrato"] = 3
    datos.loc[datos["estrato"] == "ABC1", "estrato"] = 5
    datos.loc[datos["estrato"] == "C2", "estrato"] = 4
    datos.loc[datos["estrato"] == "C3", "estrato"] = 3
    datos.loc[datos["estrato"] == "D", "estrato"] = 2
    datos.loc[datos["estrato"] == "E", "estrato"] = 1

    ab = 1415000
    c1a = (808000 + 1414000) / 6.0
    c1b = (461000 + 460000) / 6.0
    abc1 = ab * 0.4038 + c1a * 0.2885 + c1b * 0.3076
    c2 = (259000 + 460000) / 6.0
    c3 = (135000 + 258000) / 6.0
    d = (670000 + 134000) / 6.0
    e = 66000 / 3.0

    datos.loc[datos["estrato"] == 5, "id"] = abc1 * datos["cant_personas_fam"]
    datos.loc[datos["estrato"] == 4, "id"] = c2 * datos["cant_personas_fam"]
    datos.loc[datos["estrato"] == 3, "id"] = c3 * datos["cant_personas_fam"]
    datos.loc[datos["estrato"] == 2, "id"] = d * datos["cant_personas_fam"]
    datos.loc[datos["estrato"] == 1, "id"] = e * datos["cant_personas_fam"]

    datos = datos.rename(columns={'id': 'ingreso_mensual'})

    datos = datos.loc[:, selectOpt]

    # se indica que son columnas enteras
    for name in selectOpt:
        # print(name)
        datos[name] = datos[name].astype('int')

    return datos


def validacion_cruzada(data, neuronas, ciclos, activacion, optimizacion,
                       grafico):
    """
    Genera el modelo de MLP y procede a realizar el entrenamiento
    y posterior test del modelo
    :param data: dataframe con datos
    :param neuronas: cantidad de neuronas capa oculta
    :param ciclos: numero de ciclos de MLP
    :param activacion: Escoger funcion de activacion. 0: relu, 1: tanh
    :param optimizacion: Escoger funcion de optimizacion de pesos. 0: lbfgs, 1: sgd
    :param grafico: Desea realizar el grafico del error. 0: no, 1: si
    :return: 
    """
    if (activacion == 0):
        activacion = 'relu'
    elif (activacion == 1):
        activacion = 'tanh'

    if (optimizacion == 0):
        optimizacion = 'lbfgs'
    elif (optimizacion == 1):
        optimizacion = 'sgd'

    print(
    "Una MLP con %i neuronas, funcion activacion: %s , metodo optimizacion pesos: %s, numero de iteraciones: %i" % (
    neuronas, activacion, optimizacion, ciclos))
    clf = MLPClassifier(solver=optimizacion, activation=activacion, alpha=1e-5,
                        hidden_layer_sizes=(neuronas,), random_state=1,
                        max_iter=ciclos)
    x_train, x_test, y_train, y_test = train_test_split(
        data.loc[:, "ingreso_mensual":], data['status_salud_publica'], test_size=0.3)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("Accuracy: %.2f " % (score * 100))
    print("Loss: %.2f " % (clf.loss_ * 100))
    if (grafico == 1):
        # graficar
        plt.plot(clf.loss_curve_)
        plt.title(
            "MLP con %i neuronas, accuracy: %.2f" % (neuronas, score * 100))
        plt.ylabel("Error")
        plt.xlabel("Iteraciones")
        plt.show()


parser = argparse.ArgumentParser(
    description='Ejecucion del dataSet BigData con MLP')
parser.add_argument('-n', '--neuronas',
                    help='Numero de neuronas en la capa oculta', required=True,
                    type=int)
parser.add_argument('-c', '--ciclos', help='Numero de epocas de la MLP',
                    required=True, type=int)
parser.add_argument('-a', '--activacion',
                    help='Escoger funcion de activacion. 0: relu, 1: tanh',
                    required=True, type=int)
parser.add_argument('-s', '--optimizacion',
                    help='Escoger funcion de optimizacion de pesos. 0: lbfgs, 1: sgd',
                    required=True, type=int)
parser.add_argument('-g', '--grafico',
                    help='Indica si se desea el grafico de error. 0: no, 1: si',
                    required=True, type=int)

args = parser.parse_args()

data = leer_database()
validacion_cruzada(data, args.neuronas, args.ciclos, args.activacion,
                   args.optimizacion, args.grafico)

