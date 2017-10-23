from sklearn.ensemble import RandomForestClassifier #para implementar el random forest
import numpy as np
import pandas
from sklearn import cross_validation
from sklearn.model_selection import train_test_split

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
for name in selectOpt[1:len(selectOpt)]:
    #print(name
    datos[name] = datos[name].astype('float')

#datos.loc[:, 'est_civil':] = (datos.loc[:, 'est_civil':] - datos.loc[:, 'est_civil':].mean()) / (datos.loc[:, 'est_civil':].max() - datos.loc[:, 'est_civil':].min())


datos = datos.reset_index(drop=True)

 
# random seed 
np.random.seed(3) 

#activation function
def lineal(x):                                        
    return x

def tanh_array(x):
    return np.tanh(x)


# Helper function to plot a decision boundary. 
# If you don't fully understand this function don't worry, it just generates the contour plot below. 
def plot_decision_boundary(pred_func): 
    # Set min and max values and give it some padding 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    h = 0.01 
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    #print(xx)
    #print(yy)
    #print(np.c_[xx.ravel(), yy.ravel()])
    # Predict the function value for the whole gid 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    #print(Z)
    Z = Z.reshape(xx.shape) 
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 

#print("num_examples",num_examples)
#print("nn_input_dim", nn_input_dim)
#print("nn_output_dim", nn_output_dim)
 
# Gradient descent parameters (I picked these by hand) 
epsilon = 0.01 # learning rate for gradient descent 
reg_lambda = 0.01 # regularization strength 

# Helper function to evaluate the total loss on the dataset 
def calculate_loss(model,X,y, n, e,num_examples): 
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 
    # Forward propagation to calculate our predictions 
    z1 = X.dot(W1) + b1
    if(n == 0):
        a1 = lineal(z1)
        #print a1
    elif(n == 1):
        a1 = tanh_array(z1)
    z2 = a1.dot(W2) + b2 
    #print "z2",z2
    exp_scores = np.exp(z2) 
    #print "es",exp_scores
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
    # Calculating the loss 
    if(e == 0):
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs) 
    elif(e == 1):
        #print probs.shape
        final = probs[range(num_examples), y]
        errores = np.power(final-1,2)
        data_loss = np.sum(errores)/num_examples
        return data_loss
        #print errores
    # Add regulatization term to loss (optional) 
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2))) 
    return 1./num_examples * data_loss 

# Helper function to predict an output (0 or 1) 
def predict(model, x, n): 
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 
    # Forward propagation 
    z1 = x.dot(W1) + b1 
    if(n == 0):
        a1 = lineal(z1)
    elif(n == 1):
        a1 = tanh_array(z1)
    print(a1)
    print(z1)
    z2 = a1.dot(W2) + b2 
    exp_scores = np.exp(z2) 
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
    return np.argmax(probs, axis=1) 

# This function learns parameters for the neural network and returns the model. 
# - nn_hdim: Number of nodes in the hidden layer 
# - X: data generate model
# - y: data predictions model
# - n: type function activation
# - e: type function error
# - num_examples: number the examples 
# - num_passes: Number of passes through the training data for gradient descent 
# - print_loss: If True, print the loss every 1000 iterations 
def build_model(nn_hdim,X,y,n,e, num_examples, nn_input_dim,nn_output_dim,  num_passes=1, print_loss=False): 
 
    # Initialize the parameters to random values. We need to learn these. 
    np.random.seed(0) 
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim) 
    b1 = np.zeros((1, nn_hdim)) 
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim) 
    b2 = np.zeros((1, nn_output_dim)) 

    #print("W1 shape", W1.shape)
    #print("b1 shape", b1.shape)
    #print("W2 shape", W2.shape)
    #print("b2 shape", b2.shape)
 
    # This is what we return at the end 
    model = {} 
    
    #guardar el error
    errores = []
 
    # Gradient descent. For each batch... 
    for i in range(0, num_passes): 
        print(i)
        # Forward propagation 
        z1 = X.dot(W1) + b1 
        if(n == 0):
            a1 = lineal(z1)
        elif(n == 1):
            a1 = tanh_array(z1)
        z2 = a1.dot(W2) + b2 
        exp_scores = np.exp(z2) 
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
        #print("FORWARD STAGE")
        #print("z1 shape",z1.shape)
        #print("a1 shape",z1.shape)
        #print("z1 shape",z1.shape)
        # Backpropagation 
        delta3 = probs 
        #print "1",delta3
        delta3[range(num_examples), y] -= 1 
        #print "2",delta3
        dW2 = (a1.T).dot(delta3) 
        #print "dw2", dW2
        db2 = np.sum(delta3, axis=0, keepdims=True) 
        if(n == 0):
            delta2 = delta3.dot(W2.T)
        elif (n == 1):
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2)) 
        dW1 = np.dot(X.T, delta2) 
        db1 = np.sum(delta2, axis=0) 
 
        # Add regularization terms (b1 and b2 don't have regularization terms) 
        dW2 += reg_lambda * W2 
        dW1 += reg_lambda * W1 
 
        # Gradient descent parameter update 
        W1 += -epsilon * dW1 
        b1 += -epsilon * db1 
        W2 += -epsilon * dW2 
        b2 += -epsilon * db2 
 
        # Assign new parameters to the model 
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2} 
 
        # Optionally print the loss. 
        # This is expensive because it uses the whole dataset, so we don't want to do it too often. 
        if print_loss and i % 1000 == 0: 
            errores.append(calculate_loss(model, X,y, n, e, num_examples))
            
    return model, errores


def validacion_cruzada(data, neu, cil, n, e):
    
    print ""
    print "Comenzando con la generacion del modelo"
    print "neuronas: %i, funcion transferencia: %i, funcion objetivo: %i"%(neu,n,e)
    
    #todas las columnas
    #x_total= pandas.DataFrame.as_matrix(data.loc[:, 'est_civil':])
    #y_total = np.array(data['status_salud_publica'])

    #cv = cross_validation.KFold(len(data), n_folds=10)
    X, x_test, Y, y_test = train_test_split(data.loc[:, "est_civil":], data['status_salud_publica'], test_size=0.3)

    vp = 0
    vn = 0
    fp = 0
    fn = 0
    it = 0
    errores = []

    #se inicia la validacion cruzada
    #for traincv, testcv in cv:
        
    X = pandas.DataFrame.as_matrix(X)
    x_test = pandas.DataFrame.as_matrix(x_test)

    y = np.array(Y).astype(int)

    y_test = np.array(y_test).astype(int)

    # %% 15 
    num_examples = len(X) # training set size 
    print(num_examples)
    nn_input_dim = len(X[0]) # input layer dimensionality 
    print(nn_input_dim)
    nn_output_dim = 2 # output layer dimensionality 

    print "iteracion: %i"%(it)
    # %% 17 
    # Build a model with a 3-dimensional hidden layer 
    #for p in range
    model, err = build_model(neu,X,y, n,e, num_examples, nn_input_dim,nn_output_dim, num_passes=cil, print_loss=True) 
    #   print err
    errores.append(err)
    
    solv = predict(model, x_test, n)
    
    p = pandas.crosstab(y_test, solv, rownames=['Clase real'], colnames=['Prediccion clase'])
    
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

    it+=1

    #pro = []
    #for i in range(0,cil/1000):
    #    suma = 0
    #    for j in range(10):
    #        suma+=errores[j][i]
    #    pro.append(suma/10.0)

    #son mostrados los resultados
    print "Exactitud: ",(vp+vn)/float(vp+vn+fp+fn)
    print "%i | %i"%(vp,fp)
    print "%i | %i"%(fn,vn)

    return (vp+vn)/float(vp+vn+fp+fn)
    
#implementar cross-validation
df = pandas.DataFrame(columns=['neuronas','ciclos','n','e','exactitud'],dtype=float)

data = datos

cil = 200 #numero de ciclos
#df = df.append({'neuronas':10,'ciclos':cil,'n': 0,'e': 1,'exactitud':validacion_cruzada(data, 10, cil, 0, 1)  }, ignore_index=True)
#df = df.append({'neuronas':20,'ciclos':cil,'n': 0,'e': 0,'exactitud':validacion_cruzada(data, 20, cil, 0, 0)  }, ignore_index=True)
#df = df.append({'neuronas':30,'ciclos':cil,'n': 1,'e': 0,'exactitud':validacion_cruzada(data, 30, cil, 1, 0)  }, ignore_index=True)
#df = df.append({'neuronas':40,'ciclos':cil,'n': 0,'e': 1,'exactitud':validacion_cruzada(data, 40, cil, 0, 1)  }, ignore_index=True)
#df = df.append({'neuronas':50,'ciclos':cil,'n': 1,'e': 1,'exactitud':validacion_cruzada(data, 50, cil, 1, 1)  }, ignore_index=True)
#df = df.append({'neuronas':60,'ciclos':cil,'n': 1,'e': 0,'exactitud':validacion_cruzada(data, 60, cil, 1, 0)  }, ignore_index=True)
#df = df.append({'neuronas':70,'ciclos':cil,'n': 1,'e': 0,'exactitud':validacion_cruzada(data, 70, cil, 1, 0)  }, ignore_index=True)
#df = df.append({'neuronas':80,'ciclos':cil,'n': 1,'e': 0,'exactitud':validacion_cruzada(data, 80, cil, 1, 0)  }, ignore_index=True)
#df = df.append({'neuronas':90,'ciclos':cil,'n': 1,'e': 1,'exactitud':validacion_cruzada(data, 90, cil, 1, 1)  }, ignore_index=True)
df = df.append({'neuronas':100,'ciclos':cil,'n': 0,'e': 0,'exactitud':validacion_cruzada(data, 100, cil, 0, 0)  }, ignore_index=True)
#df = df.append({'neuronas':110,'ciclos':cil,'n': 0,'e': 1,'exactitud':validacion_cruzada(data, 110, cil, 0, 1)  }, ignore_index=True)
#df = df.append({'neuronas':120,'ciclos':cil,'n': 1,'e': 1,'exactitud':validacion_cruzada(data, 120, cil, 1, 1)  }, ignore_index=True)
#df = df.append({'neuronas':40,'ciclos':cil,'n': 0,'e': 1,'exactitud':validacion_cruzada(data, 40, cil, 0, 1)  }, ignore_index=True)
print df