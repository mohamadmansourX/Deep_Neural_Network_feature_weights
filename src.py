## embeddings=Model(<layer with weights to be studied>) 
#Work have been done on keras model
from tqdm.notebook import tqdm

def avg_act_p(ii,jj):
  M = len(x_train)
  pij = 0.0
  for k in range(M):
    aijq=activ_a_k(ii,jj,k)
    pij+= abs(aijq)
  return pij/float(M)
def activ_a_k(ir,jr,kr): #for xi = k input vector
  weights=weights1 #Can be loaded from files wights_last_layer.npy
  weights = weights[0]
  bias = weights[1]
  xi = float(x_train[kr][ir])
  wij = float(weights[ir][jr])
  bj = float(bias[jr])
  aij = wij*xi+bj
  return aij


cijk={}

for lk in tqdm(range(0,model93.layers[0].layers[-3].get_config()['units'],)):
  p=0.0
  #print("neuron:{}".format(lk))
  for iii in range(len(x_train[0])):
      p+=avg_act_p(iii,lk)
  #print("   p: {}, neuron: {}".format(p,lk))
  cijk[str(lk)] = p
  #print("Neuron {} has total avg absolute activ potential = {}".format(lk,p))
#calculating the net positive contriution of one feauture i
def netcontr(ith):
  def relcontr(irel,jrel):
    Ninp = len(x_train[0])
    numer= avg_act_p(irel,jrel)
    deno = cijk[str(jrel)]
    #print(numer/deno)
    return numer/deno
  c_plus_ith=0.0
  Nhidden_layers = model93.layers[0].layers[-3].get_config()['units']
  for jnet in range(Nhidden_layers):
    cij= relcontr(ith,jnet)
    #print("i : {}, j:{}, cij: {}".format(ith,j,cij))
    #if cij>0:
    #  c_plus_ith +=  np.log(1. + np.exp(cij))
    c_plus_ith += cij
    #print("        {}".format(c_plus_ith)
  
  #print("for feature {}:  {}".format(ith, c_plus_ith))
  return c_plus_ith
features_c={}
for features in tqdm(range(len(x_train[0]))):
  features_c[str(features)]=netcontr(features)
import operator
values = list(features_c.values()) # if we need them sorted then replace with: values=sorted(features_c.items(), key=operator.itemgetter(1), reverse=True)
plt.figure(figsize=(10, 10))
plt.bar(range(len(values)), values)
plt.ylim(10,)
plt.xlabel("Features")
plt.show()
