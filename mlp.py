from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


def leer_database():
	data = pandas.read_csv('./NO_C_2017.csv')
	data1 = pandas.read_csv('./A.csv')
	data2 = pandas.read_csv('./B.csv')
	data3 = pandas.read_csv('./C.csv')
	data4 = pandas.read_csv('./D.csv')

	#unir 
	datos = pandas.concat([data, data1, data2, data3, data4])

	#mesclar los datos
	datos = datos.sample(frac = 1)
	datos = datos.sample(frac = 1)
	datos = datos.sample(frac = 1)

	datos = datos.reset_index(drop=True)

	selectOpt = ["status_salud_publica","est_civil","sexo_desc","edad","estrato",
	"ind_interd","comuna","ind_region_rm","avaluo_bbrr","cant_bbrr","avaluo_auto","cant_autos",
	"n_actividad","n_rubros","clean2","tot_docs","tot_mont","ind_morosidad1","ind_morosidad2","ind_consultas_id",
	"cant_personas_fam","cant_hijos_fam"]

	datos = datos.loc[:,selectOpt]

	datos.loc[((datos["est_civil"] == " ") & (datos["cant_hijos_fam"] == 0)), "est_civil"] = "SOLTERO"
	datos.loc[((datos["est_civil"] == " ") & (datos["cant_hijos_fam"] > 0)), "est_civil"] = "CASADO"

	#son pasados los estados a una variable binaria
	datos.loc[datos["status_salud_publica"] == "S", "status_salud_publica"]= 1
	datos.loc[datos["status_salud_publica"] == "N", "status_salud_publica"]= 0

	#cambair sexo de la persona
	datos.loc[datos["sexo_desc"] == "F", "sexo_desc"]= 0
	datos.loc[datos["sexo_desc"] == "M", "sexo_desc"]= 1
	datos.loc[datos["sexo_desc"] == "SI" , "sexo_desc"] = 0

	#ind_interd
	datos.loc[datos["ind_interd"] == " " , "ind_interd"] = 0
	datos.loc[datos["ind_interd"] == "N" , "ind_interd"] = 0
	datos.loc[datos["ind_interd"] == "S" , "ind_interd"] = 1

	#gente de fuera de santiago agregar comuna
	datos.loc[datos["est_civil"] == "SOLTERO", "est_civil"]= 0
	datos.loc[datos["est_civil"] == "CASADO", "est_civil"]= 3
	datos.loc[datos["est_civil"] == "VIUDO", "est_civil"]= 1
	datos.loc[datos["est_civil"] == "DIVORCIADO", "est_civil"]= 2
	datos.loc[datos["est_civil"] == "SEPARADO JUDICIALMENTE", "est_civil"]= 2

	#son discretizados los estratos sociales
	datos.loc[datos["estrato"] == "SIN CLASIFICACION", "estrato"]= 3
	datos.loc[datos["estrato"] == "ABC1", "estrato"]= 5
	datos.loc[datos["estrato"] == "C2", "estrato"]= 4
	datos.loc[datos["estrato"] == "C3", "estrato"]= 3
	datos.loc[datos["estrato"] == "D", "estrato"]= 2
	datos.loc[datos["estrato"] == "E", "estrato"]= 1

	#se indica que son columnas enteras
	for name in selectOpt:
	    #print(name)
	    datos[name] = datos[name].astype('int')
 
	return datos

def validacion_cruzada(data, neu, cil, activation_o, solver_o):
    
    #cv = cross_validation.KFold(len(data), n_folds=10)
    X, x_test, Y, y_test = train_test_split(data.loc[:, "est_civil":], data['status_salud_publica'], test_size=0.3)

    vp = 0
    vn = 0
    fp = 0
    fn = 0
        
    X = pandas.DataFrame.as_matrix(X)
    x_test = pandas.DataFrame.as_matrix(x_test)

    y = np.array(Y).astype(int)

    y_test = np.array(y_test).astype(int)

    if(activation_o == 0):
    	activ = 'relu'
    elif(activation_o == 1):
    	activ = 'tanh'

    if(solver_o == 0):
    	solv = 'adam'
    elif(solver_o == 1):
    	solv = 'sgd'

    ##print("una MLP con %i neuronas, funcion activacion: %s , metodo minimizar: %s, numero de iteraciones: %i"%(neu,activ,solv,cil))
	
	clf = MLPClassifier(solver=solv, activation = activ, alpha=1e-5, hidden_layer_sizes=(neu, ), random_state=1)
	#cv = ShuffleSplit(n_splits=2, test_size=0.3, random_state=0)
	scores = cross_val_score(clf, data.loc[:, "est_civil":], data['status_salud_publica'], cv=3)
	print scores
	#clf.fit(X,y)
	
	#solucion = clf.predict(x_test)
	"""
    p = pandas.crosstab(y_test, solucion, rownames=['Clase real'], colnames=['Prediccion clase'])
    
    try:
        vp += p[0][0]
        fn += p[0][1]
        fp += p[1][0]
        vn += p[1][1]
    except:
        vp += p[0][0]
        fp += 0
        fn += p[0][1]
        vn += 0
    """
    #son mostrados los resultados
    print "Exactitud: ",(vp+vn)/float(vp+vn+fp+fn+1)
    print "%i | %i"%(vp,fp)
    print "%i | %i"%(fn,vn)

    return (vp+vn)/float(vp+vn+fp+fn+1)


data = leer_database()

df = pandas.DataFrame(columns=['neuronas','ciclos','n','e','exactitud'],dtype=float)


cil = 300 #numero de ciclos
df = df.append({'neuronas':10,'ciclos':cil,'n': 0,'e': 1,'exactitud':validacion_cruzada(data, 10, cil, 1, 0)  }, ignore_index=True)
#df = df.append({'neuronas':20,'ciclos':cil,'n': 0,'e': 0,'exactitud':validacion_cruzada(data, 20, cil, 0, 0)  }, ignore_index=True)
#df = df.append({'neuronas':30,'ciclos':cil,'n': 1,'e': 0,'exactitud':validacion_cruzada(data, 30, cil, 1, 0)  }, ignore_index=True)
#df = df.append({'neuronas':40,'ciclos':cil,'n': 0,'e': 1,'exactitud':validacion_cruzada(data, 40, cil, 0, 1)  }, ignore_index=True)
#df = df.append({'neuronas':50,'ciclos':cil,'n': 1,'e': 1,'exactitud':validacion_cruzada(data, 50, cil, 1, 1)  }, ignore_index=True)
#df = df.append({'neuronas':60,'ciclos':cil,'n': 1,'e': 0,'exactitud':validacion_cruzada(data, 60, cil, 1, 0)  }, ignore_index=True)
#df = df.append({'neuronas':70,'ciclos':cil,'n': 1,'e': 0,'exactitud':validacion_cruzada(data, 70, cil, 1, 0)  }, ignore_index=True)
#df = df.append({'neuronas':80,'ciclos':cil,'n': 1,'e': 0,'exactitud':validacion_cruzada(data, 80, cil, 1, 0)  }, ignore_index=True)
#df = df.append({'neuronas':90,'ciclos':cil,'n': 1,'e': 1,'exactitud':validacion_cruzada(data, 90, cil, 1, 1)  }, ignore_index=True)
#df = df.append({'neuronas':150,'ciclos':cil,'n': 0,'e': 0,'exactitud':validacion_cruzada(data, 150, cil, 0, 0)  }, ignore_index=True)
#df = df.append({'neuronas':110,'ciclos':cil,'n': 0,'e': 1,'exactitud':validacion_cruzada(data, 110, cil, 0, 1)  }, ignore_index=True)
#df = df.append({'neuronas':220,'ciclos':cil,'n': 1,'e': 1,'exactitud':validacion_cruzada(data, 220, cil, 1, 1)  }, ignore_index=True)
#df = df.append({'neuronas':200,'ciclos':cil,'n': 0,'e': 1,'exactitud':validacion_cruzada(data, 200, cil, 0, 1)  }, ignore_index=True)

print df
