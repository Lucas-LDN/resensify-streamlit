Ć
ŃŁ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878łť	

text_model/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:şî*0
shared_name!text_model/embedding/embeddings

3text_model/embedding/embeddings/Read/ReadVariableOpReadVariableOptext_model/embedding/embeddings*!
_output_shapes
:şî*
dtype0

text_model/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nametext_model/conv1d/kernel

,text_model/conv1d/kernel/Read/ReadVariableOpReadVariableOptext_model/conv1d/kernel*$
_output_shapes
:*
dtype0

text_model/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nametext_model/conv1d/bias
~
*text_model/conv1d/bias/Read/ReadVariableOpReadVariableOptext_model/conv1d/bias*
_output_shapes	
:*
dtype0

text_model/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametext_model/conv1d_1/kernel

.text_model/conv1d_1/kernel/Read/ReadVariableOpReadVariableOptext_model/conv1d_1/kernel*$
_output_shapes
:*
dtype0

text_model/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nametext_model/conv1d_1/bias

,text_model/conv1d_1/bias/Read/ReadVariableOpReadVariableOptext_model/conv1d_1/bias*
_output_shapes	
:*
dtype0

text_model/conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametext_model/conv1d_2/kernel

.text_model/conv1d_2/kernel/Read/ReadVariableOpReadVariableOptext_model/conv1d_2/kernel*$
_output_shapes
:*
dtype0

text_model/conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nametext_model/conv1d_2/bias

,text_model/conv1d_2/bias/Read/ReadVariableOpReadVariableOptext_model/conv1d_2/bias*
_output_shapes	
:*
dtype0

text_model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nametext_model/dense/kernel

+text_model/dense/kernel/Read/ReadVariableOpReadVariableOptext_model/dense/kernel* 
_output_shapes
:
*
dtype0

text_model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nametext_model/dense/bias
|
)text_model/dense/bias/Read/ReadVariableOpReadVariableOptext_model/dense/bias*
_output_shapes	
:*
dtype0

text_model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_nametext_model/dense_1/kernel

-text_model/dense_1/kernel/Read/ReadVariableOpReadVariableOptext_model/dense_1/kernel*
_output_shapes
:	*
dtype0

text_model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nametext_model/dense_1/bias

+text_model/dense_1/bias/Read/ReadVariableOpReadVariableOptext_model/dense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Ť
&Adam/text_model/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:şî*7
shared_name(&Adam/text_model/embedding/embeddings/m
¤
:Adam/text_model/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp&Adam/text_model/embedding/embeddings/m*!
_output_shapes
:şî*
dtype0
 
Adam/text_model/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/text_model/conv1d/kernel/m

3Adam/text_model/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/text_model/conv1d/kernel/m*$
_output_shapes
:*
dtype0

Adam/text_model/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/text_model/conv1d/bias/m

1Adam/text_model/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/text_model/conv1d/bias/m*
_output_shapes	
:*
dtype0
¤
!Adam/text_model/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/text_model/conv1d_1/kernel/m

5Adam/text_model/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/text_model/conv1d_1/kernel/m*$
_output_shapes
:*
dtype0

Adam/text_model/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/text_model/conv1d_1/bias/m

3Adam/text_model/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/text_model/conv1d_1/bias/m*
_output_shapes	
:*
dtype0
¤
!Adam/text_model/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/text_model/conv1d_2/kernel/m

5Adam/text_model/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/text_model/conv1d_2/kernel/m*$
_output_shapes
:*
dtype0

Adam/text_model/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/text_model/conv1d_2/bias/m

3Adam/text_model/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/text_model/conv1d_2/bias/m*
_output_shapes	
:*
dtype0

Adam/text_model/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/text_model/dense/kernel/m

2Adam/text_model/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/text_model/dense/kernel/m* 
_output_shapes
:
*
dtype0

Adam/text_model/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/text_model/dense/bias/m

0Adam/text_model/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/text_model/dense/bias/m*
_output_shapes	
:*
dtype0

 Adam/text_model/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/text_model/dense_1/kernel/m

4Adam/text_model/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/text_model/dense_1/kernel/m*
_output_shapes
:	*
dtype0

Adam/text_model/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/text_model/dense_1/bias/m

2Adam/text_model/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/text_model/dense_1/bias/m*
_output_shapes
:*
dtype0
Ť
&Adam/text_model/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:şî*7
shared_name(&Adam/text_model/embedding/embeddings/v
¤
:Adam/text_model/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp&Adam/text_model/embedding/embeddings/v*!
_output_shapes
:şî*
dtype0
 
Adam/text_model/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/text_model/conv1d/kernel/v

3Adam/text_model/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/text_model/conv1d/kernel/v*$
_output_shapes
:*
dtype0

Adam/text_model/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/text_model/conv1d/bias/v

1Adam/text_model/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/text_model/conv1d/bias/v*
_output_shapes	
:*
dtype0
¤
!Adam/text_model/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/text_model/conv1d_1/kernel/v

5Adam/text_model/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/text_model/conv1d_1/kernel/v*$
_output_shapes
:*
dtype0

Adam/text_model/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/text_model/conv1d_1/bias/v

3Adam/text_model/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/text_model/conv1d_1/bias/v*
_output_shapes	
:*
dtype0
¤
!Adam/text_model/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/text_model/conv1d_2/kernel/v

5Adam/text_model/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/text_model/conv1d_2/kernel/v*$
_output_shapes
:*
dtype0

Adam/text_model/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/text_model/conv1d_2/bias/v

3Adam/text_model/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/text_model/conv1d_2/bias/v*
_output_shapes	
:*
dtype0

Adam/text_model/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/text_model/dense/kernel/v

2Adam/text_model/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/text_model/dense/kernel/v* 
_output_shapes
:
*
dtype0

Adam/text_model/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/text_model/dense/bias/v

0Adam/text_model/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/text_model/dense/bias/v*
_output_shapes	
:*
dtype0

 Adam/text_model/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/text_model/dense_1/kernel/v

4Adam/text_model/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/text_model/dense_1/kernel/v*
_output_shapes
:	*
dtype0

Adam/text_model/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/text_model/dense_1/bias/v

2Adam/text_model/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/text_model/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Í?
valueĂ?BŔ? Bš?
ä
	embedding

cnn_layer1

cnn_layer2

cnn_layer3
pool
dense_1
dropout

last_dense
		optimizer

trainable_variables
	variables
regularization_losses
	keras_api

signatures
b

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

 kernel
!bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
R
&trainable_variables
'	variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
R
0trainable_variables
1	variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api

:iter

;beta_1

<beta_2
	=decay
>learning_ratemwmxmymzm{ m|!m}*m~+m4m5mvvvvv v!v*v+v4v5v
N
0
1
2
3
4
 5
!6
*7
+8
49
510
N
0
1
2
3
4
 5
!6
*7
+8
49
510
 
­
?layer_metrics

trainable_variables
@non_trainable_variables
	variables
Alayer_regularization_losses
regularization_losses

Blayers
Cmetrics
 
db
VARIABLE_VALUEtext_model/embedding/embeddings/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
Dlayer_metrics
trainable_variables
Enon_trainable_variables
	variables
Flayer_regularization_losses
regularization_losses

Glayers
Hmetrics
ZX
VARIABLE_VALUEtext_model/conv1d/kernel,cnn_layer1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEtext_model/conv1d/bias*cnn_layer1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Ilayer_metrics
trainable_variables
Jnon_trainable_variables
	variables
Klayer_regularization_losses
regularization_losses

Llayers
Mmetrics
\Z
VARIABLE_VALUEtext_model/conv1d_1/kernel,cnn_layer2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEtext_model/conv1d_1/bias*cnn_layer2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Nlayer_metrics
trainable_variables
Onon_trainable_variables
	variables
Player_regularization_losses
regularization_losses

Qlayers
Rmetrics
\Z
VARIABLE_VALUEtext_model/conv1d_2/kernel,cnn_layer3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEtext_model/conv1d_2/bias*cnn_layer3/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
­
Slayer_metrics
"trainable_variables
Tnon_trainable_variables
#	variables
Ulayer_regularization_losses
$regularization_losses

Vlayers
Wmetrics
 
 
 
­
Xlayer_metrics
&trainable_variables
Ynon_trainable_variables
'	variables
Zlayer_regularization_losses
(regularization_losses

[layers
\metrics
VT
VARIABLE_VALUEtext_model/dense/kernel)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEtext_model/dense/bias'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
­
]layer_metrics
,trainable_variables
^non_trainable_variables
-	variables
_layer_regularization_losses
.regularization_losses

`layers
ametrics
 
 
 
­
blayer_metrics
0trainable_variables
cnon_trainable_variables
1	variables
dlayer_regularization_losses
2regularization_losses

elayers
fmetrics
[Y
VARIABLE_VALUEtext_model/dense_1/kernel,last_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtext_model/dense_1/bias*last_dense/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
­
glayer_metrics
6trainable_variables
hnon_trainable_variables
7	variables
ilayer_regularization_losses
8regularization_losses

jlayers
kmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
8
0
1
2
3
4
5
6
7

l0
m1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ntotal
	ocount
p	variables
q	keras_api
D
	rtotal
	scount
t
_fn_kwargs
u	variables
v	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

n0
o1

p	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

r0
s1

u	variables

VARIABLE_VALUE&Adam/text_model/embedding/embeddings/mKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/text_model/conv1d/kernel/mHcnn_layer1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/text_model/conv1d/bias/mFcnn_layer1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE!Adam/text_model/conv1d_1/kernel/mHcnn_layer2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/text_model/conv1d_1/bias/mFcnn_layer2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE!Adam/text_model/conv1d_2/kernel/mHcnn_layer3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/text_model/conv1d_2/bias/mFcnn_layer3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/text_model/dense/kernel/mEdense_1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/text_model/dense/bias/mCdense_1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/text_model/dense_1/kernel/mHlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/text_model/dense_1/bias/mFlast_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/text_model/embedding/embeddings/vKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/text_model/conv1d/kernel/vHcnn_layer1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/text_model/conv1d/bias/vFcnn_layer1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE!Adam/text_model/conv1d_1/kernel/vHcnn_layer2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/text_model/conv1d_1/bias/vFcnn_layer2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE!Adam/text_model/conv1d_2/kernel/vHcnn_layer3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/text_model/conv1d_2/bias/vFcnn_layer3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/text_model/dense/kernel/vEdense_1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/text_model/dense/bias/vCdense_1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE Adam/text_model/dense_1/kernel/vHlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/text_model/dense_1/bias/vFlast_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ę
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1text_model/embedding/embeddingstext_model/conv1d/kerneltext_model/conv1d/biastext_model/conv1d_1/kerneltext_model/conv1d_1/biastext_model/conv1d_2/kerneltext_model/conv1d_2/biastext_model/dense/kerneltext_model/dense/biastext_model/dense_1/kerneltext_model/dense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_34257
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ô
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3text_model/embedding/embeddings/Read/ReadVariableOp,text_model/conv1d/kernel/Read/ReadVariableOp*text_model/conv1d/bias/Read/ReadVariableOp.text_model/conv1d_1/kernel/Read/ReadVariableOp,text_model/conv1d_1/bias/Read/ReadVariableOp.text_model/conv1d_2/kernel/Read/ReadVariableOp,text_model/conv1d_2/bias/Read/ReadVariableOp+text_model/dense/kernel/Read/ReadVariableOp)text_model/dense/bias/Read/ReadVariableOp-text_model/dense_1/kernel/Read/ReadVariableOp+text_model/dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp:Adam/text_model/embedding/embeddings/m/Read/ReadVariableOp3Adam/text_model/conv1d/kernel/m/Read/ReadVariableOp1Adam/text_model/conv1d/bias/m/Read/ReadVariableOp5Adam/text_model/conv1d_1/kernel/m/Read/ReadVariableOp3Adam/text_model/conv1d_1/bias/m/Read/ReadVariableOp5Adam/text_model/conv1d_2/kernel/m/Read/ReadVariableOp3Adam/text_model/conv1d_2/bias/m/Read/ReadVariableOp2Adam/text_model/dense/kernel/m/Read/ReadVariableOp0Adam/text_model/dense/bias/m/Read/ReadVariableOp4Adam/text_model/dense_1/kernel/m/Read/ReadVariableOp2Adam/text_model/dense_1/bias/m/Read/ReadVariableOp:Adam/text_model/embedding/embeddings/v/Read/ReadVariableOp3Adam/text_model/conv1d/kernel/v/Read/ReadVariableOp1Adam/text_model/conv1d/bias/v/Read/ReadVariableOp5Adam/text_model/conv1d_1/kernel/v/Read/ReadVariableOp3Adam/text_model/conv1d_1/bias/v/Read/ReadVariableOp5Adam/text_model/conv1d_2/kernel/v/Read/ReadVariableOp3Adam/text_model/conv1d_2/bias/v/Read/ReadVariableOp2Adam/text_model/dense/kernel/v/Read/ReadVariableOp0Adam/text_model/dense/bias/v/Read/ReadVariableOp4Adam/text_model/dense_1/kernel/v/Read/ReadVariableOp2Adam/text_model/dense_1/bias/v/Read/ReadVariableOpConst*7
Tin0
.2,	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_34761
§
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametext_model/embedding/embeddingstext_model/conv1d/kerneltext_model/conv1d/biastext_model/conv1d_1/kerneltext_model/conv1d_1/biastext_model/conv1d_2/kerneltext_model/conv1d_2/biastext_model/dense/kerneltext_model/dense/biastext_model/dense_1/kerneltext_model/dense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1&Adam/text_model/embedding/embeddings/mAdam/text_model/conv1d/kernel/mAdam/text_model/conv1d/bias/m!Adam/text_model/conv1d_1/kernel/mAdam/text_model/conv1d_1/bias/m!Adam/text_model/conv1d_2/kernel/mAdam/text_model/conv1d_2/bias/mAdam/text_model/dense/kernel/mAdam/text_model/dense/bias/m Adam/text_model/dense_1/kernel/mAdam/text_model/dense_1/bias/m&Adam/text_model/embedding/embeddings/vAdam/text_model/conv1d/kernel/vAdam/text_model/conv1d/bias/v!Adam/text_model/conv1d_1/kernel/vAdam/text_model/conv1d_1/bias/v!Adam/text_model/conv1d_2/kernel/vAdam/text_model/conv1d_2/bias/vAdam/text_model/dense/kernel/vAdam/text_model/dense/bias/v Adam/text_model/dense_1/kernel/vAdam/text_model/dense_1/bias/v*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_34897śű
Ü	

D__inference_embedding_layer_call_and_return_conditional_losses_34463

inputs,
(embedding_lookup_readvariableop_resource
identityŽ
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*!
_output_shapes
:şî*
dtype02!
embedding_lookup/ReadVariableOp¤
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
embedding_lookup
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
embedding_lookup/Identity
IdentityIdentity"embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ž
¨
@__inference_dense_layer_call_and_return_conditional_losses_34556

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

}
(__inference_conv1d_1_layer_call_fn_34520

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_339142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*<
_input_shapes+
):˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Řł
¤
!__inference__traced_restore_34897
file_prefix4
0assignvariableop_text_model_embedding_embeddings/
+assignvariableop_1_text_model_conv1d_kernel-
)assignvariableop_2_text_model_conv1d_bias1
-assignvariableop_3_text_model_conv1d_1_kernel/
+assignvariableop_4_text_model_conv1d_1_bias1
-assignvariableop_5_text_model_conv1d_2_kernel/
+assignvariableop_6_text_model_conv1d_2_bias.
*assignvariableop_7_text_model_dense_kernel,
(assignvariableop_8_text_model_dense_bias0
,assignvariableop_9_text_model_dense_1_kernel/
+assignvariableop_10_text_model_dense_1_bias!
assignvariableop_11_adam_iter#
assignvariableop_12_adam_beta_1#
assignvariableop_13_adam_beta_2"
assignvariableop_14_adam_decay*
&assignvariableop_15_adam_learning_rate
assignvariableop_16_total
assignvariableop_17_count
assignvariableop_18_total_1
assignvariableop_19_count_1>
:assignvariableop_20_adam_text_model_embedding_embeddings_m7
3assignvariableop_21_adam_text_model_conv1d_kernel_m5
1assignvariableop_22_adam_text_model_conv1d_bias_m9
5assignvariableop_23_adam_text_model_conv1d_1_kernel_m7
3assignvariableop_24_adam_text_model_conv1d_1_bias_m9
5assignvariableop_25_adam_text_model_conv1d_2_kernel_m7
3assignvariableop_26_adam_text_model_conv1d_2_bias_m6
2assignvariableop_27_adam_text_model_dense_kernel_m4
0assignvariableop_28_adam_text_model_dense_bias_m8
4assignvariableop_29_adam_text_model_dense_1_kernel_m6
2assignvariableop_30_adam_text_model_dense_1_bias_m>
:assignvariableop_31_adam_text_model_embedding_embeddings_v7
3assignvariableop_32_adam_text_model_conv1d_kernel_v5
1assignvariableop_33_adam_text_model_conv1d_bias_v9
5assignvariableop_34_adam_text_model_conv1d_1_kernel_v7
3assignvariableop_35_adam_text_model_conv1d_1_bias_v9
5assignvariableop_36_adam_text_model_conv1d_2_kernel_v7
3assignvariableop_37_adam_text_model_conv1d_2_bias_v6
2assignvariableop_38_adam_text_model_dense_kernel_v4
0assignvariableop_39_adam_text_model_dense_bias_v8
4assignvariableop_40_adam_text_model_dense_1_kernel_v6
2assignvariableop_41_adam_text_model_dense_1_bias_v
identity_43˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_28˘AssignVariableOp_29˘AssignVariableOp_3˘AssignVariableOp_30˘AssignVariableOp_31˘AssignVariableOp_32˘AssignVariableOp_33˘AssignVariableOp_34˘AssignVariableOp_35˘AssignVariableOp_36˘AssignVariableOp_37˘AssignVariableOp_38˘AssignVariableOp_39˘AssignVariableOp_4˘AssignVariableOp_40˘AssignVariableOp_41˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*
valueB+B/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB,cnn_layer1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*cnn_layer1/bias/.ATTRIBUTES/VARIABLE_VALUEB,cnn_layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB*cnn_layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB,cnn_layer3/kernel/.ATTRIBUTES/VARIABLE_VALUEB*cnn_layer3/bias/.ATTRIBUTES/VARIABLE_VALUEB)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUEB,last_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB*last_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFcnn_layer1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFcnn_layer2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFcnn_layer3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEdense_1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdense_1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFlast_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFcnn_layer1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFcnn_layer2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFcnn_layer3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEdense_1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCdense_1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFlast_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesä
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Â
_output_shapesŻ
Ź:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityŻ
AssignVariableOpAssignVariableOp0assignvariableop_text_model_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1°
AssignVariableOp_1AssignVariableOp+assignvariableop_1_text_model_conv1d_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ž
AssignVariableOp_2AssignVariableOp)assignvariableop_2_text_model_conv1d_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3˛
AssignVariableOp_3AssignVariableOp-assignvariableop_3_text_model_conv1d_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4°
AssignVariableOp_4AssignVariableOp+assignvariableop_4_text_model_conv1d_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5˛
AssignVariableOp_5AssignVariableOp-assignvariableop_5_text_model_conv1d_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6°
AssignVariableOp_6AssignVariableOp+assignvariableop_6_text_model_conv1d_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ż
AssignVariableOp_7AssignVariableOp*assignvariableop_7_text_model_dense_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8­
AssignVariableOp_8AssignVariableOp(assignvariableop_8_text_model_dense_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ą
AssignVariableOp_9AssignVariableOp,assignvariableop_9_text_model_dense_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ł
AssignVariableOp_10AssignVariableOp+assignvariableop_10_text_model_dense_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11Ľ
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12§
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ś
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ž
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ą
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ą
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ł
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ł
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Â
AssignVariableOp_20AssignVariableOp:assignvariableop_20_adam_text_model_embedding_embeddings_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ť
AssignVariableOp_21AssignVariableOp3assignvariableop_21_adam_text_model_conv1d_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22š
AssignVariableOp_22AssignVariableOp1assignvariableop_22_adam_text_model_conv1d_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23˝
AssignVariableOp_23AssignVariableOp5assignvariableop_23_adam_text_model_conv1d_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ť
AssignVariableOp_24AssignVariableOp3assignvariableop_24_adam_text_model_conv1d_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25˝
AssignVariableOp_25AssignVariableOp5assignvariableop_25_adam_text_model_conv1d_2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ť
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_text_model_conv1d_2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ş
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_text_model_dense_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¸
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_text_model_dense_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ź
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_text_model_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ş
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_text_model_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Â
AssignVariableOp_31AssignVariableOp:assignvariableop_31_adam_text_model_embedding_embeddings_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ť
AssignVariableOp_32AssignVariableOp3assignvariableop_32_adam_text_model_conv1d_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33š
AssignVariableOp_33AssignVariableOp1assignvariableop_33_adam_text_model_conv1d_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34˝
AssignVariableOp_34AssignVariableOp5assignvariableop_34_adam_text_model_conv1d_1_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35ť
AssignVariableOp_35AssignVariableOp3assignvariableop_35_adam_text_model_conv1d_1_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36˝
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_text_model_conv1d_2_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ť
AssignVariableOp_37AssignVariableOp3assignvariableop_37_adam_text_model_conv1d_2_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ş
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_text_model_dense_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¸
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_text_model_dense_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ź
AssignVariableOp_40AssignVariableOp4assignvariableop_40_adam_text_model_dense_1_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41ş
AssignVariableOp_41AssignVariableOp2assignvariableop_41_adam_text_model_dense_1_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_419
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpú
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_42í
Identity_43IdentityIdentity_42:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_43"#
identity_43Identity_43:output:0*ż
_input_shapes­
Ş: ::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ú
|
'__inference_dense_1_layer_call_fn_34612

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallň
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_340342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ü	

D__inference_embedding_layer_call_and_return_conditional_losses_33853

inputs,
(embedding_lookup_readvariableop_resource
identityŽ
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*!
_output_shapes
:şî*
dtype02!
embedding_lookup/ReadVariableOp¤
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
embedding_lookup
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
embedding_lookup/Identity
IdentityIdentity"embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
/

E__inference_text_model_layer_call_and_return_conditional_losses_34089
input_1
embedding_34054
conv1d_34057
conv1d_34059
conv1d_1_34063
conv1d_1_34065
conv1d_2_34069
conv1d_2_34071
dense_34077
dense_34079
dense_1_34083
dense_1_34085
identity˘conv1d/StatefulPartitionedCall˘ conv1d_1/StatefulPartitionedCall˘ conv1d_2/StatefulPartitionedCall˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘!embedding/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_34054*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_338532#
!embedding/StatefulPartitionedCallš
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_34057conv1d_34059*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_338812 
conv1d/StatefulPartitionedCall
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342&
$global_max_pooling1d/PartitionedCallĂ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_1_34063conv1d_1_34065*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_339142"
 conv1d_1/StatefulPartitionedCall
&global_max_pooling1d/PartitionedCall_1PartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342(
&global_max_pooling1d/PartitionedCall_1Ă
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_2_34069conv1d_2_34071*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_339472"
 conv1d_2/StatefulPartitionedCall
&global_max_pooling1d/PartitionedCall_2PartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342(
&global_max_pooling1d/PartitionedCall_2e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
concat/axis˙
concatConcatV2-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d/PartitionedCall_1:output:0/global_max_pooling1d/PartitionedCall_2:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
concat
dense/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_34077dense_34079*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_339772
dense/StatefulPartitionedCallń
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_340102
dropout/PartitionedCallŚ
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_34083dense_1_34085*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_340342!
dense_1/StatefulPartitionedCallÉ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
î
¸
C__inference_conv1d_2_layer_call_and_return_conditional_losses_33947

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d/ExpandDimsş
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimš
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1Á
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*<
_input_shapes+
):˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

}
(__inference_conv1d_2_layer_call_fn_34545

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_339472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*<
_input_shapes+
):˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ŃQ
ţ
E__inference_text_model_layer_call_and_return_conditional_losses_34400

inputs6
2embedding_embedding_lookup_readvariableop_resource6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityĚ
)embedding/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource*!
_output_shapes
:şî*
dtype02+
)embedding/embedding_lookup/ReadVariableOpÂ
embedding/embedding_lookup/axisConst*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2!
embedding/embedding_lookup/axisĘ
embedding/embedding_lookupGatherV21embedding/embedding_lookup/ReadVariableOp:value:0inputs(embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
embedding/embedding_lookupť
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2%
#embedding/embedding_lookup/Identity
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2
conv1d/conv1d/ExpandDims/dimŰ
conv1d/conv1d/ExpandDims
ExpandDims,embedding/embedding_lookup/Identity:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d/conv1d/ExpandDimsĎ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimŐ
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/conv1d/ExpandDims_1Ý
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv1d/conv1dą
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2
conv1d/conv1d/Squeeze˘
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv1d/BiasAdd/ReadVariableOp˛
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d/BiasAdd{
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d/Relu
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*global_max_pooling1d/Max/reduction_indicesž
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
global_max_pooling1d/Max
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2 
conv1d_1/conv1d/ExpandDims/dimá
conv1d_1/conv1d/ExpandDims
ExpandDims,embedding/embedding_lookup/Identity:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d_1/conv1d/ExpandDimsŐ
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimÝ
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d_1/conv1d/ExpandDims_1ĺ
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv1d_1/conv1dˇ
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2
conv1d_1/conv1d/Squeeze¨
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOpş
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d_1/BiasAdd
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d_1/Relu
,global_max_pooling1d/Max_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,global_max_pooling1d/Max_1/reduction_indicesĆ
global_max_pooling1d/Max_1Maxconv1d_1/Relu:activations:05global_max_pooling1d/Max_1/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
global_max_pooling1d/Max_1
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2 
conv1d_2/conv1d/ExpandDims/dimá
conv1d_2/conv1d/ExpandDims
ExpandDims,embedding/embedding_lookup/Identity:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d_2/conv1d/ExpandDimsŐ
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimÝ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d_2/conv1d/ExpandDims_1ĺ
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv1d_2/conv1dˇ
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2
conv1d_2/conv1d/Squeeze¨
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOpş
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d_2/BiasAdd
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d_2/Relu
,global_max_pooling1d/Max_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,global_max_pooling1d/Max_2/reduction_indicesĆ
global_max_pooling1d/Max_2Maxconv1d_2/Relu:activations:05global_max_pooling1d/Max_2/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
global_max_pooling1d/Max_2e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
concat/axisŰ
concatConcatV2!global_max_pooling1d/Max:output:0#global_max_pooling1d/Max_1:output:0#global_max_pooling1d/Max_2:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
concatĄ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulconcat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

dense/Relu}
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/IdentityŚ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpĄ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/Softmaxm
IdentityIdentitydense_1/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::::::::::::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
î
¸
C__inference_conv1d_1_layer_call_and_return_conditional_losses_34511

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d/ExpandDimsş
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimš
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1Á
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*<
_input_shapes+
):˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
/

E__inference_text_model_layer_call_and_return_conditional_losses_34195

inputs
embedding_34160
conv1d_34163
conv1d_34165
conv1d_1_34169
conv1d_1_34171
conv1d_2_34175
conv1d_2_34177
dense_34183
dense_34185
dense_1_34189
dense_1_34191
identity˘conv1d/StatefulPartitionedCall˘ conv1d_1/StatefulPartitionedCall˘ conv1d_2/StatefulPartitionedCall˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘!embedding/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_34160*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_338532#
!embedding/StatefulPartitionedCallš
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_34163conv1d_34165*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_338812 
conv1d/StatefulPartitionedCall
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342&
$global_max_pooling1d/PartitionedCallĂ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_1_34169conv1d_1_34171*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_339142"
 conv1d_1/StatefulPartitionedCall
&global_max_pooling1d/PartitionedCall_1PartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342(
&global_max_pooling1d/PartitionedCall_1Ă
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_2_34175conv1d_2_34177*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_339472"
 conv1d_2/StatefulPartitionedCall
&global_max_pooling1d/PartitionedCall_2PartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342(
&global_max_pooling1d/PartitionedCall_2e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
concat/axis˙
concatConcatV2-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d/PartitionedCall_1:output:0/global_max_pooling1d/PartitionedCall_2:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
concat
dense/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_34183dense_34185*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_339772
dense/StatefulPartitionedCallń
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_340102
dropout/PartitionedCallŚ
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_34189dense_1_34191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_340342!
dense_1/StatefulPartitionedCallÉ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
č

#__inference_signature_wrapper_34257
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity˘StatefulPartitionedCallĆ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_338272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
î
¸
C__inference_conv1d_2_layer_call_and_return_conditional_losses_34536

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d/ExpandDimsş
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimš
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1Á
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*<
_input_shapes+
):˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ć
P
4__inference_global_max_pooling1d_layer_call_fn_33840

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:e a
=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	

*__inference_text_model_layer_call_fn_34220
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity˘StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_text_model_layer_call_and_return_conditional_losses_341952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
	

*__inference_text_model_layer_call_fn_34427

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity˘StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_text_model_layer_call_and_return_conditional_losses_341302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
É
`
B__inference_dropout_layer_call_and_return_conditional_losses_34582

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

{
&__inference_conv1d_layer_call_fn_34495

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall˙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_338812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*<
_input_shapes+
):˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

`
'__inference_dropout_layer_call_fn_34587

inputs
identity˘StatefulPartitionedCallŮ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_340052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ž
¨
@__inference_dense_layer_call_and_return_conditional_losses_33977

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	

*__inference_text_model_layer_call_fn_34454

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity˘StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_text_model_layer_call_and_return_conditional_losses_341952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ę
o
)__inference_embedding_layer_call_fn_34470

inputs
unknown
identity˘StatefulPartitionedCallő
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_338532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˝0
Ľ
E__inference_text_model_layer_call_and_return_conditional_losses_34051
input_1
embedding_33862
conv1d_33892
conv1d_33894
conv1d_1_33925
conv1d_1_33927
conv1d_2_33958
conv1d_2_33960
dense_33988
dense_33990
dense_1_34045
dense_1_34047
identity˘conv1d/StatefulPartitionedCall˘ conv1d_1/StatefulPartitionedCall˘ conv1d_2/StatefulPartitionedCall˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘dropout/StatefulPartitionedCall˘!embedding/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_33862*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_338532#
!embedding/StatefulPartitionedCallš
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_33892conv1d_33894*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_338812 
conv1d/StatefulPartitionedCall
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342&
$global_max_pooling1d/PartitionedCallĂ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_1_33925conv1d_1_33927*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_339142"
 conv1d_1/StatefulPartitionedCall
&global_max_pooling1d/PartitionedCall_1PartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342(
&global_max_pooling1d/PartitionedCall_1Ă
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_2_33958conv1d_2_33960*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_339472"
 conv1d_2/StatefulPartitionedCall
&global_max_pooling1d/PartitionedCall_2PartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342(
&global_max_pooling1d/PartitionedCall_2e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
concat/axis˙
concatConcatV2-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d/PartitionedCall_1:output:0/global_max_pooling1d/PartitionedCall_2:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
concat
dense/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_33988dense_33990*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_339772
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_340052!
dropout/StatefulPartitionedCallŽ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_34045dense_1_34047*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_340342!
dense_1/StatefulPartitionedCallë
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1

a
B__inference_dropout_layer_call_and_return_conditional_losses_34577

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeľ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>2
dropout/GreaterEqual/yż
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ř
z
%__inference_dense_layer_call_fn_34565

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallń
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_339772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˛
Ş
B__inference_dense_1_layer_call_and_return_conditional_losses_34603

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
î
¸
C__inference_conv1d_1_layer_call_and_return_conditional_losses_33914

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d/ExpandDimsş
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimš
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1Á
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*<
_input_shapes+
):˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

a
B__inference_dropout_layer_call_and_return_conditional_losses_34005

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeľ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>2
dropout/GreaterEqual/yż
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ť[
é
__inference__traced_save_34761
file_prefix>
:savev2_text_model_embedding_embeddings_read_readvariableop7
3savev2_text_model_conv1d_kernel_read_readvariableop5
1savev2_text_model_conv1d_bias_read_readvariableop9
5savev2_text_model_conv1d_1_kernel_read_readvariableop7
3savev2_text_model_conv1d_1_bias_read_readvariableop9
5savev2_text_model_conv1d_2_kernel_read_readvariableop7
3savev2_text_model_conv1d_2_bias_read_readvariableop6
2savev2_text_model_dense_kernel_read_readvariableop4
0savev2_text_model_dense_bias_read_readvariableop8
4savev2_text_model_dense_1_kernel_read_readvariableop6
2savev2_text_model_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopE
Asavev2_adam_text_model_embedding_embeddings_m_read_readvariableop>
:savev2_adam_text_model_conv1d_kernel_m_read_readvariableop<
8savev2_adam_text_model_conv1d_bias_m_read_readvariableop@
<savev2_adam_text_model_conv1d_1_kernel_m_read_readvariableop>
:savev2_adam_text_model_conv1d_1_bias_m_read_readvariableop@
<savev2_adam_text_model_conv1d_2_kernel_m_read_readvariableop>
:savev2_adam_text_model_conv1d_2_bias_m_read_readvariableop=
9savev2_adam_text_model_dense_kernel_m_read_readvariableop;
7savev2_adam_text_model_dense_bias_m_read_readvariableop?
;savev2_adam_text_model_dense_1_kernel_m_read_readvariableop=
9savev2_adam_text_model_dense_1_bias_m_read_readvariableopE
Asavev2_adam_text_model_embedding_embeddings_v_read_readvariableop>
:savev2_adam_text_model_conv1d_kernel_v_read_readvariableop<
8savev2_adam_text_model_conv1d_bias_v_read_readvariableop@
<savev2_adam_text_model_conv1d_1_kernel_v_read_readvariableop>
:savev2_adam_text_model_conv1d_1_bias_v_read_readvariableop@
<savev2_adam_text_model_conv1d_2_kernel_v_read_readvariableop>
:savev2_adam_text_model_conv1d_2_bias_v_read_readvariableop=
9savev2_adam_text_model_dense_kernel_v_read_readvariableop;
7savev2_adam_text_model_dense_bias_v_read_readvariableop?
;savev2_adam_text_model_dense_1_kernel_v_read_readvariableop=
9savev2_adam_text_model_dense_1_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ee692919f81e4518ae0fac5476edf5bd/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*
valueB+B/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB,cnn_layer1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*cnn_layer1/bias/.ATTRIBUTES/VARIABLE_VALUEB,cnn_layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB*cnn_layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB,cnn_layer3/kernel/.ATTRIBUTES/VARIABLE_VALUEB*cnn_layer3/bias/.ATTRIBUTES/VARIABLE_VALUEB)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUEB,last_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB*last_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFcnn_layer1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFcnn_layer2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFcnn_layer3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEdense_1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdense_1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFlast_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFcnn_layer1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFcnn_layer2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHcnn_layer3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFcnn_layer3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEdense_1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCdense_1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHlast_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFlast_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesŢ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesş
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_text_model_embedding_embeddings_read_readvariableop3savev2_text_model_conv1d_kernel_read_readvariableop1savev2_text_model_conv1d_bias_read_readvariableop5savev2_text_model_conv1d_1_kernel_read_readvariableop3savev2_text_model_conv1d_1_bias_read_readvariableop5savev2_text_model_conv1d_2_kernel_read_readvariableop3savev2_text_model_conv1d_2_bias_read_readvariableop2savev2_text_model_dense_kernel_read_readvariableop0savev2_text_model_dense_bias_read_readvariableop4savev2_text_model_dense_1_kernel_read_readvariableop2savev2_text_model_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopAsavev2_adam_text_model_embedding_embeddings_m_read_readvariableop:savev2_adam_text_model_conv1d_kernel_m_read_readvariableop8savev2_adam_text_model_conv1d_bias_m_read_readvariableop<savev2_adam_text_model_conv1d_1_kernel_m_read_readvariableop:savev2_adam_text_model_conv1d_1_bias_m_read_readvariableop<savev2_adam_text_model_conv1d_2_kernel_m_read_readvariableop:savev2_adam_text_model_conv1d_2_bias_m_read_readvariableop9savev2_adam_text_model_dense_kernel_m_read_readvariableop7savev2_adam_text_model_dense_bias_m_read_readvariableop;savev2_adam_text_model_dense_1_kernel_m_read_readvariableop9savev2_adam_text_model_dense_1_bias_m_read_readvariableopAsavev2_adam_text_model_embedding_embeddings_v_read_readvariableop:savev2_adam_text_model_conv1d_kernel_v_read_readvariableop8savev2_adam_text_model_conv1d_bias_v_read_readvariableop<savev2_adam_text_model_conv1d_1_kernel_v_read_readvariableop:savev2_adam_text_model_conv1d_1_bias_v_read_readvariableop<savev2_adam_text_model_conv1d_2_kernel_v_read_readvariableop:savev2_adam_text_model_conv1d_2_bias_v_read_readvariableop9savev2_adam_text_model_dense_kernel_v_read_readvariableop7savev2_adam_text_model_dense_bias_v_read_readvariableop;savev2_adam_text_model_dense_1_kernel_v_read_readvariableop9savev2_adam_text_model_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+	2
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesű
ř: :şî:::::::
::	:: : : : : : : : : :şî:::::::
::	::şî:::::::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:'#
!
_output_shapes
:şî:*&
$
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
:!	

_output_shapes	
::%
!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :'#
!
_output_shapes
:şî:*&
$
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::' #
!
_output_shapes
:şî:*!&
$
_output_shapes
::!"

_output_shapes	
::*#&
$
_output_shapes
::!$

_output_shapes	
::*%&
$
_output_shapes
::!&

_output_shapes	
::&'"
 
_output_shapes
:
:!(

_output_shapes	
::%)!

_output_shapes
:	: *

_output_shapes
::+

_output_shapes
: 

C
'__inference_dropout_layer_call_fn_34592

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_340102
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

k
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_33834

inputs
identityp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:e a
=
_output_shapes+
):'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ě
ś
A__inference_conv1d_layer_call_and_return_conditional_losses_33881

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d/ExpandDimsş
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimš
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1Á
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*<
_input_shapes+
):˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ě
ś
A__inference_conv1d_layer_call_and_return_conditional_losses_34486

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d/ExpandDimsş
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimš
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1Á
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*<
_input_shapes+
):˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Śa
Ó
 __inference__wrapped_model_33827
input_1A
=text_model_embedding_embedding_lookup_readvariableop_resourceA
=text_model_conv1d_conv1d_expanddims_1_readvariableop_resource5
1text_model_conv1d_biasadd_readvariableop_resourceC
?text_model_conv1d_1_conv1d_expanddims_1_readvariableop_resource7
3text_model_conv1d_1_biasadd_readvariableop_resourceC
?text_model_conv1d_2_conv1d_expanddims_1_readvariableop_resource7
3text_model_conv1d_2_biasadd_readvariableop_resource3
/text_model_dense_matmul_readvariableop_resource4
0text_model_dense_biasadd_readvariableop_resource5
1text_model_dense_1_matmul_readvariableop_resource6
2text_model_dense_1_biasadd_readvariableop_resource
identityí
4text_model/embedding/embedding_lookup/ReadVariableOpReadVariableOp=text_model_embedding_embedding_lookup_readvariableop_resource*!
_output_shapes
:şî*
dtype026
4text_model/embedding/embedding_lookup/ReadVariableOpă
*text_model/embedding/embedding_lookup/axisConst*G
_class=
;9loc:@text_model/embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2,
*text_model/embedding/embedding_lookup/axis
%text_model/embedding/embedding_lookupGatherV2<text_model/embedding/embedding_lookup/ReadVariableOp:value:0input_13text_model/embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*G
_class=
;9loc:@text_model/embedding/embedding_lookup/ReadVariableOp*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2'
%text_model/embedding/embedding_lookupÜ
.text_model/embedding/embedding_lookup/IdentityIdentity.text_model/embedding/embedding_lookup:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙20
.text_model/embedding/embedding_lookup/Identity
'text_model/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2)
'text_model/conv1d/conv1d/ExpandDims/dim
#text_model/conv1d/conv1d/ExpandDims
ExpandDims7text_model/embedding/embedding_lookup/Identity:output:00text_model/conv1d/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2%
#text_model/conv1d/conv1d/ExpandDimsđ
4text_model/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=text_model_conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype026
4text_model/conv1d/conv1d/ExpandDims_1/ReadVariableOp
)text_model/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)text_model/conv1d/conv1d/ExpandDims_1/dim
%text_model/conv1d/conv1d/ExpandDims_1
ExpandDims<text_model/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:02text_model/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2'
%text_model/conv1d/conv1d/ExpandDims_1
text_model/conv1d/conv1dConv2D,text_model/conv1d/conv1d/ExpandDims:output:0.text_model/conv1d/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
text_model/conv1d/conv1dŇ
 text_model/conv1d/conv1d/SqueezeSqueeze!text_model/conv1d/conv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2"
 text_model/conv1d/conv1d/SqueezeĂ
(text_model/conv1d/BiasAdd/ReadVariableOpReadVariableOp1text_model_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(text_model/conv1d/BiasAdd/ReadVariableOpŢ
text_model/conv1d/BiasAddBiasAdd)text_model/conv1d/conv1d/Squeeze:output:00text_model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
text_model/conv1d/BiasAdd
text_model/conv1d/ReluRelu"text_model/conv1d/BiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
text_model/conv1d/Relu°
5text_model/global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :27
5text_model/global_max_pooling1d/Max/reduction_indicesę
#text_model/global_max_pooling1d/MaxMax$text_model/conv1d/Relu:activations:0>text_model/global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#text_model/global_max_pooling1d/MaxĄ
)text_model/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2+
)text_model/conv1d_1/conv1d/ExpandDims/dim
%text_model/conv1d_1/conv1d/ExpandDims
ExpandDims7text_model/embedding/embedding_lookup/Identity:output:02text_model/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2'
%text_model/conv1d_1/conv1d/ExpandDimsö
6text_model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?text_model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype028
6text_model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
+text_model/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+text_model/conv1d_1/conv1d/ExpandDims_1/dim
'text_model/conv1d_1/conv1d/ExpandDims_1
ExpandDims>text_model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:04text_model/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2)
'text_model/conv1d_1/conv1d/ExpandDims_1
text_model/conv1d_1/conv1dConv2D.text_model/conv1d_1/conv1d/ExpandDims:output:00text_model/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
text_model/conv1d_1/conv1dŘ
"text_model/conv1d_1/conv1d/SqueezeSqueeze#text_model/conv1d_1/conv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2$
"text_model/conv1d_1/conv1d/SqueezeÉ
*text_model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp3text_model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*text_model/conv1d_1/BiasAdd/ReadVariableOpć
text_model/conv1d_1/BiasAddBiasAdd+text_model/conv1d_1/conv1d/Squeeze:output:02text_model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
text_model/conv1d_1/BiasAdd˘
text_model/conv1d_1/ReluRelu$text_model/conv1d_1/BiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
text_model/conv1d_1/Relu´
7text_model/global_max_pooling1d/Max_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7text_model/global_max_pooling1d/Max_1/reduction_indicesň
%text_model/global_max_pooling1d/Max_1Max&text_model/conv1d_1/Relu:activations:0@text_model/global_max_pooling1d/Max_1/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%text_model/global_max_pooling1d/Max_1Ą
)text_model/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2+
)text_model/conv1d_2/conv1d/ExpandDims/dim
%text_model/conv1d_2/conv1d/ExpandDims
ExpandDims7text_model/embedding/embedding_lookup/Identity:output:02text_model/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2'
%text_model/conv1d_2/conv1d/ExpandDimsö
6text_model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?text_model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype028
6text_model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
+text_model/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+text_model/conv1d_2/conv1d/ExpandDims_1/dim
'text_model/conv1d_2/conv1d/ExpandDims_1
ExpandDims>text_model/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:04text_model/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2)
'text_model/conv1d_2/conv1d/ExpandDims_1
text_model/conv1d_2/conv1dConv2D.text_model/conv1d_2/conv1d/ExpandDims:output:00text_model/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
text_model/conv1d_2/conv1dŘ
"text_model/conv1d_2/conv1d/SqueezeSqueeze#text_model/conv1d_2/conv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2$
"text_model/conv1d_2/conv1d/SqueezeÉ
*text_model/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp3text_model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*text_model/conv1d_2/BiasAdd/ReadVariableOpć
text_model/conv1d_2/BiasAddBiasAdd+text_model/conv1d_2/conv1d/Squeeze:output:02text_model/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
text_model/conv1d_2/BiasAdd˘
text_model/conv1d_2/ReluRelu$text_model/conv1d_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
text_model/conv1d_2/Relu´
7text_model/global_max_pooling1d/Max_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7text_model/global_max_pooling1d/Max_2/reduction_indicesň
%text_model/global_max_pooling1d/Max_2Max&text_model/conv1d_2/Relu:activations:0@text_model/global_max_pooling1d/Max_2/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%text_model/global_max_pooling1d/Max_2{
text_model/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
text_model/concat/axis
text_model/concatConcatV2,text_model/global_max_pooling1d/Max:output:0.text_model/global_max_pooling1d/Max_1:output:0.text_model/global_max_pooling1d/Max_2:output:0text_model/concat/axis:output:0*
N*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
text_model/concatÂ
&text_model/dense/MatMul/ReadVariableOpReadVariableOp/text_model_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02(
&text_model/dense/MatMul/ReadVariableOpť
text_model/dense/MatMulMatMultext_model/concat:output:0.text_model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
text_model/dense/MatMulŔ
'text_model/dense/BiasAdd/ReadVariableOpReadVariableOp0text_model_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'text_model/dense/BiasAdd/ReadVariableOpĆ
text_model/dense/BiasAddBiasAdd!text_model/dense/MatMul:product:0/text_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
text_model/dense/BiasAdd
text_model/dense/ReluRelu!text_model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
text_model/dense/Relu
text_model/dropout/IdentityIdentity#text_model/dense/Relu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
text_model/dropout/IdentityÇ
(text_model/dense_1/MatMul/ReadVariableOpReadVariableOp1text_model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(text_model/dense_1/MatMul/ReadVariableOpĘ
text_model/dense_1/MatMulMatMul$text_model/dropout/Identity:output:00text_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
text_model/dense_1/MatMulĹ
)text_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp2text_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)text_model/dense_1/BiasAdd/ReadVariableOpÍ
text_model/dense_1/BiasAddBiasAdd#text_model/dense_1/MatMul:product:01text_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
text_model/dense_1/BiasAdd
text_model/dense_1/SoftmaxSoftmax#text_model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
text_model/dense_1/Softmaxx
IdentityIdentity$text_model/dense_1/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::::::::::::Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
˛
Ş
B__inference_dense_1_layer_call_and_return_conditional_losses_34034

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ş0
¤
E__inference_text_model_layer_call_and_return_conditional_losses_34130

inputs
embedding_34095
conv1d_34098
conv1d_34100
conv1d_1_34104
conv1d_1_34106
conv1d_2_34110
conv1d_2_34112
dense_34118
dense_34120
dense_1_34124
dense_1_34126
identity˘conv1d/StatefulPartitionedCall˘ conv1d_1/StatefulPartitionedCall˘ conv1d_2/StatefulPartitionedCall˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘dropout/StatefulPartitionedCall˘!embedding/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_34095*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_338532#
!embedding/StatefulPartitionedCallš
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_34098conv1d_34100*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_338812 
conv1d/StatefulPartitionedCall
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342&
$global_max_pooling1d/PartitionedCallĂ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_1_34104conv1d_1_34106*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_339142"
 conv1d_1/StatefulPartitionedCall
&global_max_pooling1d/PartitionedCall_1PartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342(
&global_max_pooling1d/PartitionedCall_1Ă
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_2_34110conv1d_2_34112*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_339472"
 conv1d_2/StatefulPartitionedCall
&global_max_pooling1d/PartitionedCall_2PartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_338342(
&global_max_pooling1d/PartitionedCall_2e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
concat/axis˙
concatConcatV2-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d/PartitionedCall_1:output:0/global_max_pooling1d/PartitionedCall_2:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
concat
dense/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_34118dense_34120*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_339772
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_340052!
dropout/StatefulPartitionedCallŽ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_34124dense_1_34126*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_340342!
dense_1/StatefulPartitionedCallë
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	

*__inference_text_model_layer_call_fn_34155
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity˘StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_text_model_layer_call_and_return_conditional_losses_341302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
É
`
B__inference_dropout_layer_call_and_return_conditional_losses_34010

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ÓZ
ţ
E__inference_text_model_layer_call_and_return_conditional_losses_34332

inputs6
2embedding_embedding_lookup_readvariableop_resource6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityĚ
)embedding/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource*!
_output_shapes
:şî*
dtype02+
)embedding/embedding_lookup/ReadVariableOpÂ
embedding/embedding_lookup/axisConst*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2!
embedding/embedding_lookup/axisĘ
embedding/embedding_lookupGatherV21embedding/embedding_lookup/ReadVariableOp:value:0inputs(embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
embedding/embedding_lookupť
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2%
#embedding/embedding_lookup/Identity
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2
conv1d/conv1d/ExpandDims/dimŰ
conv1d/conv1d/ExpandDims
ExpandDims,embedding/embedding_lookup/Identity:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d/conv1d/ExpandDimsĎ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimŐ
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/conv1d/ExpandDims_1Ý
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv1d/conv1dą
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2
conv1d/conv1d/Squeeze˘
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv1d/BiasAdd/ReadVariableOp˛
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d/BiasAdd{
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d/Relu
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*global_max_pooling1d/Max/reduction_indicesž
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
global_max_pooling1d/Max
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2 
conv1d_1/conv1d/ExpandDims/dimá
conv1d_1/conv1d/ExpandDims
ExpandDims,embedding/embedding_lookup/Identity:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d_1/conv1d/ExpandDimsŐ
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimÝ
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d_1/conv1d/ExpandDims_1ĺ
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv1d_1/conv1dˇ
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2
conv1d_1/conv1d/Squeeze¨
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv1d_1/BiasAdd/ReadVariableOpş
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d_1/BiasAdd
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d_1/Relu
,global_max_pooling1d/Max_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,global_max_pooling1d/Max_1/reduction_indicesĆ
global_max_pooling1d/Max_1Maxconv1d_1/Relu:activations:05global_max_pooling1d/Max_1/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
global_max_pooling1d/Max_1
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ý˙˙˙˙˙˙˙˙2 
conv1d_2/conv1d/ExpandDims/dimá
conv1d_2/conv1d/ExpandDims
ExpandDims,embedding/embedding_lookup/Identity:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d_2/conv1d/ExpandDimsŐ
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimÝ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d_2/conv1d/ExpandDims_1ĺ
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv1d_2/conv1dˇ
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
squeeze_dims

ý˙˙˙˙˙˙˙˙2
conv1d_2/conv1d/Squeeze¨
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOpş
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d_2/BiasAdd
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
conv1d_2/Relu
,global_max_pooling1d/Max_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,global_max_pooling1d/Max_2/reduction_indicesĆ
global_max_pooling1d/Max_2Maxconv1d_2/Relu:activations:05global_max_pooling1d/Max_2/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
global_max_pooling1d/Max_2e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
concat/axisŰ
concatConcatV2!global_max_pooling1d/Max:output:0#global_max_pooling1d/Max_1:output:0#global_max_pooling1d/Max_2:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
concatĄ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulconcat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/dropout/Const
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÍ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>2 
dropout/dropout/GreaterEqual/yß
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dropout/dropout/Mul_1Ś
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpĄ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/Softmaxm
IdentityIdentitydense_1/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::::::::::::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*´
serving_default 
D
input_19
serving_default_input_1:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙<
output_10
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:ŃÓ
Â
	embedding

cnn_layer1

cnn_layer2

cnn_layer3
pool
dense_1
dropout

last_dense
		optimizer

trainable_variables
	variables
regularization_losses
	keras_api

signatures
+&call_and_return_all_conditional_losses
_default_save_signature
__call__"
_tf_keras_modelç{"class_name": "BertClassifier", "name": "text_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "BertClassifier"}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["sparse_categorical_accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
°

embeddings
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerő{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 30522, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}}
é	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Â
_tf_keras_layer¨{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 256]}}
í	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ć
_tf_keras_layerŹ{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 256]}}
í	

 kernel
!bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
+&call_and_return_all_conditional_losses
__call__"Ć
_tf_keras_layerŹ{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 256]}}

&trainable_variables
'	variables
(regularization_losses
)	keras_api
+&call_and_return_all_conditional_losses
__call__"ř
_tf_keras_layerŢ{"class_name": "GlobalMaxPooling1D", "name": "global_max_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_max_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ń

*kernel
+bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
+&call_and_return_all_conditional_losses
__call__"Ę
_tf_keras_layer°{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 384}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 384]}}
ă
0trainable_variables
1	variables
2regularization_losses
3	keras_api
+&call_and_return_all_conditional_losses
__call__"Ň
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
ö

4kernel
5bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
+&call_and_return_all_conditional_losses
__call__"Ď
_tf_keras_layerľ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
Ś
:iter

;beta_1

<beta_2
	=decay
>learning_ratemwmxmymzm{ m|!m}*m~+m4m5mvvvvv v!v*v+v4v5v"
	optimizer
n
0
1
2
3
4
 5
!6
*7
+8
49
510"
trackable_list_wrapper
n
0
1
2
3
4
 5
!6
*7
+8
49
510"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
?layer_metrics

trainable_variables
@non_trainable_variables
	variables
Alayer_regularization_losses
regularization_losses

Blayers
Cmetrics
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
 serving_default"
signature_map
4:2şî2text_model/embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Dlayer_metrics
trainable_variables
Enon_trainable_variables
	variables
Flayer_regularization_losses
regularization_losses

Glayers
Hmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
0:.2text_model/conv1d/kernel
%:#2text_model/conv1d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Ilayer_metrics
trainable_variables
Jnon_trainable_variables
	variables
Klayer_regularization_losses
regularization_losses

Llayers
Mmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
2:02text_model/conv1d_1/kernel
':%2text_model/conv1d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Nlayer_metrics
trainable_variables
Onon_trainable_variables
	variables
Player_regularization_losses
regularization_losses

Qlayers
Rmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
2:02text_model/conv1d_2/kernel
':%2text_model/conv1d_2/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Slayer_metrics
"trainable_variables
Tnon_trainable_variables
#	variables
Ulayer_regularization_losses
$regularization_losses

Vlayers
Wmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Xlayer_metrics
&trainable_variables
Ynon_trainable_variables
'	variables
Zlayer_regularization_losses
(regularization_losses

[layers
\metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)
2text_model/dense/kernel
$:"2text_model/dense/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
]layer_metrics
,trainable_variables
^non_trainable_variables
-	variables
_layer_regularization_losses
.regularization_losses

`layers
ametrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
blayer_metrics
0trainable_variables
cnon_trainable_variables
1	variables
dlayer_regularization_losses
2regularization_losses

elayers
fmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
,:*	2text_model/dense_1/kernel
%:#2text_model/dense_1/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
°
glayer_metrics
6trainable_variables
hnon_trainable_variables
7	variables
ilayer_regularization_losses
8regularization_losses

jlayers
kmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ť
	ntotal
	ocount
p	variables
q	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Ź
	rtotal
	scount
t
_fn_kwargs
u	variables
v	keras_api"ĺ
_tf_keras_metricĘ{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
n0
o1"
trackable_list_wrapper
-
p	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
r0
s1"
trackable_list_wrapper
-
u	variables"
_generic_user_object
9:7şî2&Adam/text_model/embedding/embeddings/m
5:32Adam/text_model/conv1d/kernel/m
*:(2Adam/text_model/conv1d/bias/m
7:52!Adam/text_model/conv1d_1/kernel/m
,:*2Adam/text_model/conv1d_1/bias/m
7:52!Adam/text_model/conv1d_2/kernel/m
,:*2Adam/text_model/conv1d_2/bias/m
0:.
2Adam/text_model/dense/kernel/m
):'2Adam/text_model/dense/bias/m
1:/	2 Adam/text_model/dense_1/kernel/m
*:(2Adam/text_model/dense_1/bias/m
9:7şî2&Adam/text_model/embedding/embeddings/v
5:32Adam/text_model/conv1d/kernel/v
*:(2Adam/text_model/conv1d/bias/v
7:52!Adam/text_model/conv1d_1/kernel/v
,:*2Adam/text_model/conv1d_1/bias/v
7:52!Adam/text_model/conv1d_2/kernel/v
,:*2Adam/text_model/conv1d_2/bias/v
0:.
2Adam/text_model/dense/kernel/v
):'2Adam/text_model/dense/bias/v
1:/	2 Adam/text_model/dense_1/kernel/v
*:(2Adam/text_model/dense_1/bias/v
Ň2Ď
E__inference_text_model_layer_call_and_return_conditional_losses_34400
E__inference_text_model_layer_call_and_return_conditional_losses_34332
E__inference_text_model_layer_call_and_return_conditional_losses_34089
E__inference_text_model_layer_call_and_return_conditional_losses_34051°
§˛Ł
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ç2ä
 __inference__wrapped_model_33827ż
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ */˘,
*'
input_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ć2ă
*__inference_text_model_layer_call_fn_34454
*__inference_text_model_layer_call_fn_34155
*__inference_text_model_layer_call_fn_34427
*__inference_text_model_layer_call_fn_34220°
§˛Ł
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
î2ë
D__inference_embedding_layer_call_and_return_conditional_losses_34463˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ó2Đ
)__inference_embedding_layer_call_fn_34470˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ë2č
A__inference_conv1d_layer_call_and_return_conditional_losses_34486˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Đ2Í
&__inference_conv1d_layer_call_fn_34495˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_conv1d_1_layer_call_and_return_conditional_losses_34511˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ň2Ď
(__inference_conv1d_1_layer_call_fn_34520˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_conv1d_2_layer_call_and_return_conditional_losses_34536˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ň2Ď
(__inference_conv1d_2_layer_call_fn_34545˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ş2§
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_33834Ó
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *3˘0
.+'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2
4__inference_global_max_pooling1d_layer_call_fn_33840Ó
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *3˘0
.+'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ę2ç
@__inference_dense_layer_call_and_return_conditional_losses_34556˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ď2Ě
%__inference_dense_layer_call_fn_34565˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Â2ż
B__inference_dropout_layer_call_and_return_conditional_losses_34577
B__inference_dropout_layer_call_and_return_conditional_losses_34582´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
2
'__inference_dropout_layer_call_fn_34592
'__inference_dropout_layer_call_fn_34587´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ě2é
B__inference_dense_1_layer_call_and_return_conditional_losses_34603˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ń2Î
'__inference_dense_1_layer_call_fn_34612˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2B0
#__inference_signature_wrapper_34257input_1Ą
 __inference__wrapped_model_33827} !*+459˘6
/˘,
*'
input_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙ż
C__inference_conv1d_1_layer_call_and_return_conditional_losses_34511x=˘:
3˘0
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "3˘0
)&
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
(__inference_conv1d_1_layer_call_fn_34520k=˘:
3˘0
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "&#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ż
C__inference_conv1d_2_layer_call_and_return_conditional_losses_34536x !=˘:
3˘0
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "3˘0
)&
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
(__inference_conv1d_2_layer_call_fn_34545k !=˘:
3˘0
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "&#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˝
A__inference_conv1d_layer_call_and_return_conditional_losses_34486x=˘:
3˘0
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "3˘0
)&
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
&__inference_conv1d_layer_call_fn_34495k=˘:
3˘0
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "&#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ł
B__inference_dense_1_layer_call_and_return_conditional_losses_34603]450˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 {
'__inference_dense_1_layer_call_fn_34612P450˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙˘
@__inference_dense_layer_call_and_return_conditional_losses_34556^*+0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 z
%__inference_dense_layer_call_fn_34565Q*+0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙¤
B__inference_dropout_layer_call_and_return_conditional_losses_34577^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ¤
B__inference_dropout_layer_call_and_return_conditional_losses_34582^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 |
'__inference_dropout_layer_call_fn_34587Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙|
'__inference_dropout_layer_call_fn_34592Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙ş
D__inference_embedding_layer_call_and_return_conditional_losses_34463r8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "3˘0
)&
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
)__inference_embedding_layer_call_fn_34470e8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "&#˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ę
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_33834wE˘B
;˘8
63
inputs'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ˘
4__inference_global_max_pooling1d_layer_call_fn_33840jE˘B
;˘8
63
inputs'˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙°
#__inference_signature_wrapper_34257 !*+45D˘A
˘ 
:Ş7
5
input_1*'
input_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙ź
E__inference_text_model_layer_call_and_return_conditional_losses_34051s !*+45=˘:
3˘0
*'
input_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ź
E__inference_text_model_layer_call_and_return_conditional_losses_34089s !*+45=˘:
3˘0
*'
input_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ť
E__inference_text_model_layer_call_and_return_conditional_losses_34332r !*+45<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ť
E__inference_text_model_layer_call_and_return_conditional_losses_34400r !*+45<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
*__inference_text_model_layer_call_fn_34155f !*+45=˘:
3˘0
*'
input_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙
*__inference_text_model_layer_call_fn_34220f !*+45=˘:
3˘0
*'
input_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙
*__inference_text_model_layer_call_fn_34427e !*+45<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙
*__inference_text_model_layer_call_fn_34454e !*+45<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙