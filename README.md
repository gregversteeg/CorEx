#Correlation Explanation (CorEx)

The principle of *Cor*-relation *Ex*-planation has recently been introduced as a way to build rich representations that
are maximally informative about the data. This project consists of python code to build these representations.

A preliminary version of the technique is described in this paper.      
*Discovering Structure in High-Dimensional Data Through Correlation Explanation*    
Greg Ver Steeg and Aram Galstyan, NIPS 2014, http://arxiv.org/abs/1406.1222        

Some theoretical developments are described here:      
*Maximally Informative Hierarchical Representions of High-Dimensional Data*    
Greg Ver Steeg and Aram Galstyan, AISTATS 2015, http://arxiv.org/abs/1410.7404   

The code here is written by Greg Ver Steeg and Gabriel Pereyra. 

Current code implements only the techniques of the first paper. 
Additional theoretical developments appear in the second paper and more are underway. 
New functionality will be added over time. 

###Dependencies

CorEx requires numpy and scipy. If you use OS X, I recommend installing the Scipy Superpack:             
http://fonnesbeck.github.io/ScipySuperpack/

###Install

To install, download using the link on the right or clone the project by executing this command in your target directory:
```
git clone https://github.com/gregversteeg/CorEx.git
```
Use *git pull* to get updates. The code is under development. 
Please feel free to raise issues or request features using the github interface. 

## Basic Usage

### Example

```python
import corex as ce

X = np.array([[0,0,0,0,0], # A matrix with rows as samples and columns as variables.
              [0,0,0,1,1],
              [1,1,1,0,0],
              [1,1,1,1,1]], dtype=int)

layer1 = ce.Corex(n_hidden=2)  # Define the number of hidden factors to use.
layer1.fit(X)

layer1.clusters  # Each variable/column is associated with one Y_j
# array([0, 0, 0, 1, 1])
layer1.labels[0]  # Labels for each sample for Y_0
# array([0, 0, 1, 1])
layer1.labels[1]  # Labels for each sample for Y_1
# array([0, 1, 0, 1])
layer1.tcs  # TC(X;Y_j) (all info measures reported in nats). 
# array([ 1.385,  0.692])
# TC(X_Gj) >=TC(X_Gj ; Y_j)
# For this example, TC(X1,X2,X3)=1.386, TC(X4,X5) = 0.693
```

### Data format

For the basic version of CorEx, you must input a matrix of integers whose rows represent samples and whose columns
represent different variables. The values must be integers {0,1,...,k-1} where k represents the maximum number of 
values that each variable, x_i can take. By default, entries equal to -1 are treated as missing. This can be 
altered by passing a *missing_values* argument when initializing CorEx. 

### CorEx outputs

As shown in the example, *clusters* gives the variable clusters for each hidden factor Y_j and 
*labels* gives the labels for each sample for each Y_j. 
Probabilistic labels can be accessed with *p_y_given_x*. 

The total correlation explained by each hidden factor, TC(X;Y_j), is accessed with *tcs*. Outputs are sorted
so that Y_0 is always the component that explains the highest TC. 
Like point-wise mutual information, you can define point-wise total correlation measure for an individual sample, x^l     
TC(X = x^l;Y_j) == log Z_j(x)   
This quantity is accessed with *log_z*. This represents the correlations explained by Y_j for an individual sample.
A low (or even negative!) number can be obtained. This can be interpreted as a measure of how surprising an individual
observation is. This can be useful for anomaly detection. 


### Generalizations

#### Hierarchical CorEx
The simplest extension is to stack CorEx representations on top of each other. 
```
layer1 = ce.Corex(n_hidden=100)
layer2 = ce.Corex(n_hidden=10)
layer3 = ce.Corex(n_hidden=1)
Y1 = layer1.fit_transform(X)
Y2 = layer2.fit_transform(Y1)
Y3 = layer2.fit_transform(Y2)
```
The sum of total correlations explained by each layer provides a successively tighter lower bound on TC(X). 
This will be detailed in a paper in progress. To assess how large your representations should be, look at quantities
like layer.tcs. Do all the Y_j's explain some correlation (i.e., all the TCs are significantly larger than 0)? If not
you should probably use a smaller representation.

#### Missing values
You can set missing values (by specifying missing_values=-1, when calling, e.g.). CorEx is very robust to missing data.
This hasn't been extensively tested yet so be careful with this feature. (E.g., while the distribution of missing values
should not matter in principle, it does have an effect in this version.)

#### Future versions
We are currently testing extensions that allow for arbitrary data types such as continuous variables. 

## Visualization

See http://bit.ly/corexvis for examples of some of the rich visualization capabilities. 
Eventually, these will be added here. 
