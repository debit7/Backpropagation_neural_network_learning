import pandas as pd
import math as math
# b0,l1,l2,b1,h1,h2=1,0,1,1,0,0
weights_df=pd.DataFrame([['b0',1,1,0],['l1',1,-1,0],['l2',0.5,2,0],['b1',0,0,1],['h1',0,0,1.5],['h2',0,0,-1]],columns=['x','h1','h2','y'])

values_df=pd.DataFrame([['b0',1],['l1',0],['l2',1],['h1',0],['h2',0],['b1',1],['y','0'],['Ey',0],['Eh1',0],['Eh2',0]],columns=['x','values'])
initial_inputs=['b0','l1','l2']
Hidden_Layer=['b1','h1','h2']
Errors=['Ey','Eh1','Eh2']
Output=['y']
Learning_rate=0.5
def find_values(df,x1):
    df.set_index("x", inplace = True)
    val=df.loc[x1]['values']
    df.reset_index(inplace=True)
    return val
def update_values(df,x1,new_val):
    df.loc[df.x == x1,"values"] = new_val
def find_distance(df,x1,x2):
    df.set_index("x", inplace = True)
    val=df.loc[x1][x2]
    df.reset_index(inplace=True)
    return val
def update_weights(df,x1,x2,new_val):
    df.loc[df.x == x1,x2] = new_val

def step(x):
    return round(1/(1+(math.exp(x*-1))),3)
def Calculate_error(target,value):
    return round((1/2)*(target-value)**2,3)


def feed_inputs_forward(inputs,weights_df,values_df,layer,output):
    s=find_values(values_df,inputs[1])*find_distance(weights_df,inputs[1],layer[1])+\
    find_values(values_df,inputs[2])*find_distance(weights_df,inputs[2],layer[1])+\
    find_values(values_df,inputs[0])
    update_values(values_df,layer[1],step(s))
    s=find_values(values_df,inputs[2])*find_distance(weights_df,inputs[2],layer[2])+\
    find_values(values_df,inputs[1])*find_distance(weights_df,inputs[2],layer[1])+\
    find_values(values_df,inputs[0])
    update_values(values_df,layer[2],step(s))
    #output
    s=find_values(values_df,Hidden_Layer[1])*find_distance(weights_df,Hidden_Layer[1],output[0])+\
    find_values(values_df,Hidden_Layer[2])*find_distance(weights_df,Hidden_Layer[2],output[0])+\
    find_values(values_df,Hidden_Layer[0])
    # print(s)
    update_values(values_df,output[0],step(s))
    # print(Calculate_error(1,find_values(values_df,output[0])))
    pass

def back_propagation(values_df,output,target,layer,Errors):
    Ey=find_values(values_df,output[0])*(1-find_values(values_df,output[0]))*(target-find_values(values_df,output[0]))
    Ey=round(Ey,3)
    update_values(values_df,Errors[0],Ey)
    c=0
    for ly in layer[1:]:
        c+=1
        z=find_values(values_df,ly)*\
        (1-find_values(values_df,ly))*\
        (find_distance(weights_df,ly,output[0])*\
        Ey)
        z=round(z,3)
        update_values(values_df,Errors[c],z)

    pass
def Learn(Learning_rate,layer,output,values_df,Errors,weights_df,inputs):
    #update weights from hidden layer to output
    for lys in layer:
            z=find_distance(weights_df,lys,output[0])+\
                Learning_rate*\
                find_values(values_df,Errors[0])*\
                find_values(values_df,lys)
            update_weights(weights_df,lys,output[0],round(z,3))
    #update hidden layer from initial input to hidden layer
    i=0
    for lyr in layer[1:]:
        i+=1
        for inpts in inputs:
                z=find_distance(weights_df,inpts,lyr)+\
                    Learning_rate*\
                    find_values(values_df,Errors[i])*\
                    find_values(values_df,inpts)
                update_weights(weights_df,inpts,lyr,round(z,3))

    pass

feed_inputs_forward(initial_inputs,weights_df,values_df,Hidden_Layer,Output)
back_propagation(values_df,Output,1,Hidden_Layer,Errors)
Learn(Learning_rate,Hidden_Layer,Output,values_df,Errors,weights_df,initial_inputs)
print(values_df)
print(weights_df)


    
        





