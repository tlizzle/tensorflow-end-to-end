??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.22unknown8??
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

: *
dtype0
?
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_2/beta/v
?
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_2/gamma/v
?
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
: *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	? *
dtype0
?
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_1/beta/v
?
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_1/gamma/v
?
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/batch_normalization/beta/v
?
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes	
:?*
dtype0
?
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/batch_normalization/gamma/v
?
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/weekday_emb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/weekday_emb/embeddings/v
?
1Adam/weekday_emb/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/weekday_emb/embeddings/v*
_output_shapes

:*
dtype0
?
Adam/country_emb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/country_emb/embeddings/v
?
1Adam/country_emb/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/country_emb/embeddings/v*
_output_shapes

:*
dtype0
?
Adam/statezip_emb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:;2*/
shared_name Adam/statezip_emb/embeddings/v
?
2Adam/statezip_emb/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/statezip_emb/embeddings/v*
_output_shapes

:;2*
dtype0
?
Adam/city_emb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*+
shared_nameAdam/city_emb/embeddings/v
?
.Adam/city_emb/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/city_emb/embeddings/v*
_output_shapes

:-*
dtype0
?
Adam/street_emb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*-
shared_nameAdam/street_emb/embeddings/v
?
0Adam/street_emb/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/street_emb/embeddings/v*
_output_shapes

:d*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

: *
dtype0
?
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_2/beta/m
?
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_2/gamma/m
?
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
: *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	? *
dtype0
?
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_1/beta/m
?
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_1/gamma/m
?
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/batch_normalization/beta/m
?
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes	
:?*
dtype0
?
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/batch_normalization/gamma/m
?
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/weekday_emb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/weekday_emb/embeddings/m
?
1Adam/weekday_emb/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/weekday_emb/embeddings/m*
_output_shapes

:*
dtype0
?
Adam/country_emb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/country_emb/embeddings/m
?
1Adam/country_emb/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/country_emb/embeddings/m*
_output_shapes

:*
dtype0
?
Adam/statezip_emb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:;2*/
shared_name Adam/statezip_emb/embeddings/m
?
2Adam/statezip_emb/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/statezip_emb/embeddings/m*
_output_shapes

:;2*
dtype0
?
Adam/city_emb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*+
shared_nameAdam/city_emb/embeddings/m
?
.Adam/city_emb/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/city_emb/embeddings/m*
_output_shapes

:-*
dtype0
?
Adam/street_emb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*-
shared_nameAdam/street_emb/embeddings/m
?
0Adam/street_emb/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/street_emb/embeddings/m*
_output_shapes

:d*
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
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_2/moving_variance
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_2/moving_mean
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_2/beta
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
: *
dtype0
?
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_2/gamma
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	? *
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:?*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:?*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:?*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:?*
dtype0
?
weekday_emb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameweekday_emb/embeddings
?
*weekday_emb/embeddings/Read/ReadVariableOpReadVariableOpweekday_emb/embeddings*
_output_shapes

:*
dtype0
?
country_emb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_namecountry_emb/embeddings
?
*country_emb/embeddings/Read/ReadVariableOpReadVariableOpcountry_emb/embeddings*
_output_shapes

:*
dtype0
?
statezip_emb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:;2*(
shared_namestatezip_emb/embeddings
?
+statezip_emb/embeddings/Read/ReadVariableOpReadVariableOpstatezip_emb/embeddings*
_output_shapes

:;2*
dtype0
?
city_emb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*$
shared_namecity_emb/embeddings
{
'city_emb/embeddings/Read/ReadVariableOpReadVariableOpcity_emb/embeddings*
_output_shapes

:-*
dtype0
?
street_emb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_namestreet_emb/embeddings

)street_emb/embeddings/Read/ReadVariableOpReadVariableOpstreet_emb/embeddings*
_output_shapes

:d*
dtype0

NoOpNoOp
Ω
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer_with_weights-5
layer-28
layer-29
layer_with_weights-6
layer-30
 layer_with_weights-7
 layer-31
!layer-32
"layer_with_weights-8
"layer-33
#layer_with_weights-9
#layer-34
$layer-35
%layer_with_weights-10
%layer-36
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,_default_save_signature
-	optimizer
.
signatures*
* 
* 
* 
* 
* 
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5
embeddings*
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<
embeddings*
?
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
C
embeddings*
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
J
embeddings*
?
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses
Q
embeddings*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses* 
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses* 
?
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses* 
?
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses* 
?
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses* 
?
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|axis
	}gamma
~beta
moving_mean
?moving_variance*

?	keras_api* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
50
<1
C2
J3
Q4
}5
~6
7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22*
?
50
<1
C2
J3
Q4
}5
~6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
,_default_save_signature
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate5m?<m?Cm?Jm?Qm?}m?~m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?5v?<v?Cv?Jv?Qv?}v?~v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?*

?serving_default* 

50*

50*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ic
VARIABLE_VALUEstreet_emb/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

<0*

<0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ga
VARIABLE_VALUEcity_emb/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

C0*

C0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ke
VARIABLE_VALUEstatezip_emb/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

J0*

J0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
jd
VARIABLE_VALUEcountry_emb/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

Q0*

Q0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
jd
VARIABLE_VALUEweekday_emb/embeddings:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
!
}0
~1
2
?3*

}0
~1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
3
0
?1
?2
?3
?4
?5*
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
?1*
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
??
VARIABLE_VALUEAdam/street_emb/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/city_emb/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/statezip_emb/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/country_emb/embeddings/mVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/weekday_emb/embeddings/mVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_2/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_2/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/street_emb/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/city_emb/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/statezip_emb/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/country_emb/embeddings/vVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/weekday_emb/embeddings/vVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_2/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_2/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
serving_default_bathroomsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_bedroomsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
w
serving_default_cityPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
serving_default_conditionPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_countryPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_floorsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_sqft_abovePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_sqft_basementPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_sqft_livingPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_sqft_lotPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_statezipPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_streetPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
w
serving_default_viewPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_waterfrontPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_weekdayPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_yr_builtPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_yr_renovatedPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?

StatefulPartitionedCallStatefulPartitionedCallserving_default_bathroomsserving_default_bedroomsserving_default_cityserving_default_conditionserving_default_countryserving_default_floorsserving_default_sqft_aboveserving_default_sqft_basementserving_default_sqft_livingserving_default_sqft_lotserving_default_statezipserving_default_streetserving_default_viewserving_default_waterfrontserving_default_weekdayserving_default_yr_builtserving_default_yr_renovatedweekday_emb/embeddingscountry_emb/embeddingsstatezip_emb/embeddingscity_emb/embeddingsstreet_emb/embeddingsbatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization/betabatch_normalization/gamma!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebatch_normalization_1/betabatch_normalization_1/gammadense/kernel
dense/bias!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization_2/betabatch_normalization_2/gammadense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*9
_read_only_resource_inputs
 !"#$%&'*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_20752
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)street_emb/embeddings/Read/ReadVariableOp'city_emb/embeddings/Read/ReadVariableOp+statezip_emb/embeddings/Read/ReadVariableOp*country_emb/embeddings/Read/ReadVariableOp*weekday_emb/embeddings/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0Adam/street_emb/embeddings/m/Read/ReadVariableOp.Adam/city_emb/embeddings/m/Read/ReadVariableOp2Adam/statezip_emb/embeddings/m/Read/ReadVariableOp1Adam/country_emb/embeddings/m/Read/ReadVariableOp1Adam/weekday_emb/embeddings/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp0Adam/street_emb/embeddings/v/Read/ReadVariableOp.Adam/city_emb/embeddings/v/Read/ReadVariableOp2Adam/statezip_emb/embeddings/v/Read/ReadVariableOp1Adam/country_emb/embeddings/v/Read/ReadVariableOp1Adam/weekday_emb/embeddings/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*O
TinH
F2D	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_22071
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamestreet_emb/embeddingscity_emb/embeddingsstatezip_emb/embeddingscountry_emb/embeddingsweekday_emb/embeddingsbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense/kernel
dense/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/street_emb/embeddings/mAdam/city_emb/embeddings/mAdam/statezip_emb/embeddings/mAdam/country_emb/embeddings/mAdam/weekday_emb/embeddings/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/dense/kernel/mAdam/dense/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/street_emb/embeddings/vAdam/city_emb/embeddings/vAdam/statezip_emb/embeddings/vAdam/country_emb/embeddings/vAdam/weekday_emb/embeddings/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/dense/kernel/vAdam/dense/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*N
TinG
E2C*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_22279??
?	
`
D__inference_reshape_4_layer_call_and_return_conditional_losses_19882

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_21494

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19483p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_concatenate_layer_call_and_return_conditional_losses_21481
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????d:?????????-:?????????2:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:?????????
#
_user_specified_name	inputs/10:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/11:RN
'
_output_shapes
:?????????d
#
_user_specified_name	inputs/12:RN
'
_output_shapes
:?????????-
#
_user_specified_name	inputs/13:RN
'
_output_shapes
:?????????2
#
_user_specified_name	inputs/14:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/15:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/16
?	
`
D__inference_reshape_3_layer_call_and_return_conditional_losses_21421

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_19971

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21607

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_20050
	bathrooms
bedrooms
city
	condition
country

floors

sqft_above
sqft_basement
sqft_living
sqft_lot
statezip

street
view

waterfront
weekday
yr_built
yr_renovated
unknown:
	unknown_0:
	unknown_1:;2
	unknown_2:-
	unknown_3:d
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	? 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:

unknown_20:

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	bathroomsbedroomscity	conditioncountryfloors
sqft_abovesqft_basementsqft_livingsqft_lotstatezipstreetview
waterfrontweekdayyr_builtyr_renovatedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*9
_read_only_resource_inputs
 !"#$%&'*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_20001o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	bathrooms:QM
'
_output_shapes
:?????????
"
_user_specified_name
bedrooms:MI
'
_output_shapes
:?????????

_user_specified_namecity:RN
'
_output_shapes
:?????????
#
_user_specified_name	condition:PL
'
_output_shapes
:?????????
!
_user_specified_name	country:OK
'
_output_shapes
:?????????
 
_user_specified_namefloors:SO
'
_output_shapes
:?????????
$
_user_specified_name
sqft_above:VR
'
_output_shapes
:?????????
'
_user_specified_namesqft_basement:TP
'
_output_shapes
:?????????
%
_user_specified_namesqft_living:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
sqft_lot:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
statezip:OK
'
_output_shapes
:?????????
 
_user_specified_namestreet:MI
'
_output_shapes
:?????????

_user_specified_nameview:SO
'
_output_shapes
:?????????
$
_user_specified_name
waterfront:PL
'
_output_shapes
:?????????
!
_user_specified_name	weekday:QM
'
_output_shapes
:?????????
"
_user_specified_name
yr_built:UQ
'
_output_shapes
:?????????
&
_user_specified_nameyr_renovated
?
?
C__inference_city_emb_layer_call_and_return_conditional_losses_21302

inputs:
(embedding_lookup_readvariableop_resource:-
identity??embedding_lookup/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:-*
dtype0?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0Cast:y:0embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????-v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:?????????-u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:?????????-h
NoOpNoOp ^embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_statezip_emb_layer_call_fn_21309

inputs
unknown:;2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_statezip_emb_layer_call_and_return_conditional_losses_19782s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
`
D__inference_reshape_3_layer_call_and_return_conditional_losses_19868

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19565

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_20505
	bathrooms
bedrooms
city
	condition
country

floors

sqft_above
sqft_basement
sqft_living
sqft_lot
statezip

street
view

waterfront
weekday
yr_built
yr_renovated
unknown:
	unknown_0:
	unknown_1:;2
	unknown_2:-
	unknown_3:d
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	? 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:

unknown_20:

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	bathroomsbedroomscity	conditioncountryfloors
sqft_abovesqft_basementsqft_livingsqft_lotstatezipstreetview
waterfrontweekdayyr_builtyr_renovatedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*3
_read_only_resource_inputs
"#$%&'*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_20389o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	bathrooms:QM
'
_output_shapes
:?????????
"
_user_specified_name
bedrooms:MI
'
_output_shapes
:?????????

_user_specified_namecity:RN
'
_output_shapes
:?????????
#
_user_specified_name	condition:PL
'
_output_shapes
:?????????
!
_user_specified_name	country:OK
'
_output_shapes
:?????????
 
_user_specified_namefloors:SO
'
_output_shapes
:?????????
$
_user_specified_name
sqft_above:VR
'
_output_shapes
:?????????
'
_user_specified_namesqft_basement:TP
'
_output_shapes
:?????????
%
_user_specified_namesqft_living:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
sqft_lot:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
statezip:OK
'
_output_shapes
:?????????
 
_user_specified_namestreet:MI
'
_output_shapes
:?????????

_user_specified_nameview:SO
'
_output_shapes
:?????????
$
_user_specified_name
waterfront:PL
'
_output_shapes
:?????????
!
_user_specified_name	weekday:QM
'
_output_shapes
:?????????
"
_user_specified_name
yr_built:UQ
'
_output_shapes
:?????????
&
_user_specified_nameyr_renovated
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19483

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
%__inference_model_layer_call_fn_20886
inputs_bathrooms
inputs_bedrooms
inputs_city
inputs_condition
inputs_country
inputs_floors
inputs_sqft_above
inputs_sqft_basement
inputs_sqft_living
inputs_sqft_lot
inputs_statezip
inputs_street
inputs_view
inputs_waterfront
inputs_weekday
inputs_yr_built
inputs_yr_renovated
unknown:
	unknown_0:
	unknown_1:;2
	unknown_2:-
	unknown_3:d
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	? 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:

unknown_20:

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_bathroomsinputs_bedroomsinputs_cityinputs_conditioninputs_countryinputs_floorsinputs_sqft_aboveinputs_sqft_basementinputs_sqft_livinginputs_sqft_lotinputs_statezipinputs_streetinputs_viewinputs_waterfrontinputs_weekdayinputs_yr_builtinputs_yr_renovatedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*3
_read_only_resource_inputs
"#$%&'*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_20389o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_nameinputs/bathrooms:XT
'
_output_shapes
:?????????
)
_user_specified_nameinputs/bedrooms:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/city:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/condition:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/country:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/floors:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/sqft_above:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/sqft_basement:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/sqft_living:X	T
'
_output_shapes
:?????????
)
_user_specified_nameinputs/sqft_lot:X
T
'
_output_shapes
:?????????
)
_user_specified_nameinputs/statezip:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/street:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/view:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/waterfront:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/weekday:XT
'
_output_shapes
:?????????
)
_user_specified_nameinputs/yr_built:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/yr_renovated
?

+__inference_country_emb_layer_call_fn_21326

inputs
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_country_emb_layer_call_and_return_conditional_losses_19768s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_weekday_emb_layer_call_and_return_conditional_losses_21353

inputs:
(embedding_lookup_readvariableop_resource:
identity??embedding_lookup/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype0?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0Cast:y:0embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:?????????u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:?????????h
NoOpNoOp ^embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_21768

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: *
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: ?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:????????? l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: ?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:????????? h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_19938

inputs1
matmul_readvariableop_resource:	? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19530

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
*__inference_street_emb_layer_call_fn_21275

inputs
unknown:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_street_emb_layer_call_and_return_conditional_losses_19810s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_21049
inputs_bathrooms
inputs_bedrooms
inputs_city
inputs_condition
inputs_country
inputs_floors
inputs_sqft_above
inputs_sqft_basement
inputs_sqft_living
inputs_sqft_lot
inputs_statezip
inputs_street
inputs_view
inputs_waterfront
inputs_weekday
inputs_yr_built
inputs_yr_renovatedF
4weekday_emb_embedding_lookup_readvariableop_resource:F
4country_emb_embedding_lookup_readvariableop_resource:G
5statezip_emb_embedding_lookup_readvariableop_resource:;2C
1city_emb_embedding_lookup_readvariableop_resource:-E
3street_emb_embedding_lookup_readvariableop_resource:d?
0batch_normalization_cast_readvariableop_resource:	?A
2batch_normalization_cast_1_readvariableop_resource:	?A
2batch_normalization_cast_2_readvariableop_resource:	?A
2batch_normalization_cast_3_readvariableop_resource:	?A
2batch_normalization_1_cast_readvariableop_resource:	?C
4batch_normalization_1_cast_1_readvariableop_resource:	?C
4batch_normalization_1_cast_2_readvariableop_resource:	?C
4batch_normalization_1_cast_3_readvariableop_resource:	?7
$dense_matmul_readvariableop_resource:	? 3
%dense_biasadd_readvariableop_resource: @
2batch_normalization_2_cast_readvariableop_resource: B
4batch_normalization_2_cast_1_readvariableop_resource: B
4batch_normalization_2_cast_2_readvariableop_resource: B
4batch_normalization_2_cast_3_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity??'batch_normalization/Cast/ReadVariableOp?)batch_normalization/Cast_1/ReadVariableOp?)batch_normalization/Cast_2/ReadVariableOp?)batch_normalization/Cast_3/ReadVariableOp?)batch_normalization_1/Cast/ReadVariableOp?+batch_normalization_1/Cast_1/ReadVariableOp?+batch_normalization_1/Cast_2/ReadVariableOp?+batch_normalization_1/Cast_3/ReadVariableOp?)batch_normalization_2/Cast/ReadVariableOp?+batch_normalization_2/Cast_1/ReadVariableOp?+batch_normalization_2/Cast_2/ReadVariableOp?+batch_normalization_2/Cast_3/ReadVariableOp?(city_emb/embedding_lookup/ReadVariableOp?+country_emb/embedding_lookup/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?,statezip_emb/embedding_lookup/ReadVariableOp?*street_emb/embedding_lookup/ReadVariableOp?+weekday_emb/embedding_lookup/ReadVariableOpi
weekday_emb/CastCastinputs_weekday*

DstT0*

SrcT0*'
_output_shapes
:??????????
+weekday_emb/embedding_lookup/ReadVariableOpReadVariableOp4weekday_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype0?
!weekday_emb/embedding_lookup/axisConst*>
_class4
20loc:@weekday_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
weekday_emb/embedding_lookupGatherV23weekday_emb/embedding_lookup/ReadVariableOp:value:0weekday_emb/Cast:y:0*weekday_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@weekday_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:??????????
%weekday_emb/embedding_lookup/IdentityIdentity%weekday_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????i
country_emb/CastCastinputs_country*

DstT0*

SrcT0*'
_output_shapes
:??????????
+country_emb/embedding_lookup/ReadVariableOpReadVariableOp4country_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype0?
!country_emb/embedding_lookup/axisConst*>
_class4
20loc:@country_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
country_emb/embedding_lookupGatherV23country_emb/embedding_lookup/ReadVariableOp:value:0country_emb/Cast:y:0*country_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@country_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:??????????
%country_emb/embedding_lookup/IdentityIdentity%country_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????k
statezip_emb/CastCastinputs_statezip*

DstT0*

SrcT0*'
_output_shapes
:??????????
,statezip_emb/embedding_lookup/ReadVariableOpReadVariableOp5statezip_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:;2*
dtype0?
"statezip_emb/embedding_lookup/axisConst*?
_class5
31loc:@statezip_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
statezip_emb/embedding_lookupGatherV24statezip_emb/embedding_lookup/ReadVariableOp:value:0statezip_emb/Cast:y:0+statezip_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*?
_class5
31loc:@statezip_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????2?
&statezip_emb/embedding_lookup/IdentityIdentity&statezip_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????2c
city_emb/CastCastinputs_city*

DstT0*

SrcT0*'
_output_shapes
:??????????
(city_emb/embedding_lookup/ReadVariableOpReadVariableOp1city_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:-*
dtype0?
city_emb/embedding_lookup/axisConst*;
_class1
/-loc:@city_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
city_emb/embedding_lookupGatherV20city_emb/embedding_lookup/ReadVariableOp:value:0city_emb/Cast:y:0'city_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*;
_class1
/-loc:@city_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????-?
"city_emb/embedding_lookup/IdentityIdentity"city_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????-g
street_emb/CastCastinputs_street*

DstT0*

SrcT0*'
_output_shapes
:??????????
*street_emb/embedding_lookup/ReadVariableOpReadVariableOp3street_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:d*
dtype0?
 street_emb/embedding_lookup/axisConst*=
_class3
1/loc:@street_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
street_emb/embedding_lookupGatherV22street_emb/embedding_lookup/ReadVariableOp:value:0street_emb/Cast:y:0)street_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*=
_class3
1/loc:@street_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????d?
$street_emb/embedding_lookup/IdentityIdentity$street_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????dj
reshape/ShapeShape-street_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshape-street_emb/embedding_lookup/Identity:output:0reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????dj
reshape_1/ShapeShape+city_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :-?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
reshape_1/ReshapeReshape+city_emb/embedding_lookup/Identity:output:0 reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????-n
reshape_2/ShapeShape/statezip_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
reshape_2/ReshapeReshape/statezip_emb/embedding_lookup/Identity:output:0 reshape_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2m
reshape_3/ShapeShape.country_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
reshape_3/ReshapeReshape.country_emb/embedding_lookup/Identity:output:0 reshape_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????m
reshape_4/ShapeShape.weekday_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:g
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
reshape_4/ReshapeReshape.weekday_emb/embedding_lookup/Identity:output:0 reshape_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2inputs_bedroomsinputs_bathroomsinputs_sqft_livinginputs_sqft_lotinputs_floorsinputs_waterfrontinputs_viewinputs_conditioninputs_sqft_aboveinputs_sqft_basementinputs_yr_builtinputs_yr_renovatedreshape/Reshape:output:0reshape_1/Reshape:output:0reshape_2/Reshape:output:0reshape_3/Reshape:output:0reshape_4/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:??
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
#batch_normalization/batchnorm/mul_1Mulconcatenate/concat:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????u
tf.math.tanh/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:???????????
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype0?
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:??
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
%batch_normalization_1/batchnorm/mul_1Multf.math.tanh/Tanh:y:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
dense/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? h
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes
: *
dtype0?
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
: *
dtype0?
+batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0?
+batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes
: *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_2/batchnorm/addAddV23batch_normalization_2/Cast_1/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: ?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: ?
%batch_normalization_2/batchnorm/mul_1Muldropout/Identity:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:????????? ?
%batch_normalization_2/batchnorm/mul_2Mul1batch_normalization_2/Cast/ReadVariableOp:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: ?
#batch_normalization_2/batchnorm/subSub3batch_normalization_2/Cast_2/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_1/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????l
dropout_1/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp,^batch_normalization_2/Cast_2/ReadVariableOp,^batch_normalization_2/Cast_3/ReadVariableOp)^city_emb/embedding_lookup/ReadVariableOp,^country_emb/embedding_lookup/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp-^statezip_emb/embedding_lookup/ReadVariableOp+^street_emb/embedding_lookup/ReadVariableOp,^weekday_emb/embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2Z
+batch_normalization_2/Cast_2/ReadVariableOp+batch_normalization_2/Cast_2/ReadVariableOp2Z
+batch_normalization_2/Cast_3/ReadVariableOp+batch_normalization_2/Cast_3/ReadVariableOp2T
(city_emb/embedding_lookup/ReadVariableOp(city_emb/embedding_lookup/ReadVariableOp2Z
+country_emb/embedding_lookup/ReadVariableOp+country_emb/embedding_lookup/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2\
,statezip_emb/embedding_lookup/ReadVariableOp,statezip_emb/embedding_lookup/ReadVariableOp2X
*street_emb/embedding_lookup/ReadVariableOp*street_emb/embedding_lookup/ReadVariableOp2Z
+weekday_emb/embedding_lookup/ReadVariableOp+weekday_emb/embedding_lookup/ReadVariableOp:Y U
'
_output_shapes
:?????????
*
_user_specified_nameinputs/bathrooms:XT
'
_output_shapes
:?????????
)
_user_specified_nameinputs/bedrooms:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/city:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/condition:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/country:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/floors:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/sqft_above:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/sqft_basement:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/sqft_living:X	T
'
_output_shapes
:?????????
)
_user_specified_nameinputs/sqft_lot:X
T
'
_output_shapes
:?????????
)
_user_specified_nameinputs/statezip:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/street:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/view:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/waterfront:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/weekday:XT
'
_output_shapes
:?????????
)
_user_specified_nameinputs/yr_built:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/yr_renovated
?	
`
D__inference_reshape_2_layer_call_and_return_conditional_losses_19854

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
F__inference_country_emb_layer_call_and_return_conditional_losses_21336

inputs:
(embedding_lookup_readvariableop_resource:
identity??embedding_lookup/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype0?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0Cast:y:0embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:?????????u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:?????????h
NoOpNoOp ^embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_21793

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_19982`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_21666

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_19949`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
F__inference_country_emb_layer_call_and_return_conditional_losses_19768

inputs:
(embedding_lookup_readvariableop_resource:
identity??embedding_lookup/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype0?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0Cast:y:0embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:?????????u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:?????????h
NoOpNoOp ^embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_21387

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :-u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????-X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????-:S O
+
_output_shapes
:?????????-
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_21676

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_21661

inputs1
matmul_readvariableop_resource:	? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_2_layer_call_fn_21701

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19647o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_21268
inputs_bathrooms
inputs_bedrooms
inputs_city
inputs_condition
inputs_country
inputs_floors
inputs_sqft_above
inputs_sqft_basement
inputs_sqft_living
inputs_sqft_lot
inputs_statezip
inputs_street
inputs_view
inputs_waterfront
inputs_weekday
inputs_yr_built
inputs_yr_renovatedF
4weekday_emb_embedding_lookup_readvariableop_resource:F
4country_emb_embedding_lookup_readvariableop_resource:G
5statezip_emb_embedding_lookup_readvariableop_resource:;2C
1city_emb_embedding_lookup_readvariableop_resource:-E
3street_emb_embedding_lookup_readvariableop_resource:dJ
;batch_normalization_assignmovingavg_readvariableop_resource:	?L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	??
0batch_normalization_cast_readvariableop_resource:	?A
2batch_normalization_cast_1_readvariableop_resource:	?L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	?N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	?A
2batch_normalization_1_cast_readvariableop_resource:	?C
4batch_normalization_1_cast_1_readvariableop_resource:	?7
$dense_matmul_readvariableop_resource:	? 3
%dense_biasadd_readvariableop_resource: K
=batch_normalization_2_assignmovingavg_readvariableop_resource: M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource: @
2batch_normalization_2_cast_readvariableop_resource: B
4batch_normalization_2_cast_1_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity??#batch_normalization/AssignMovingAvg?2batch_normalization/AssignMovingAvg/ReadVariableOp?%batch_normalization/AssignMovingAvg_1?4batch_normalization/AssignMovingAvg_1/ReadVariableOp?'batch_normalization/Cast/ReadVariableOp?)batch_normalization/Cast_1/ReadVariableOp?%batch_normalization_1/AssignMovingAvg?4batch_normalization_1/AssignMovingAvg/ReadVariableOp?'batch_normalization_1/AssignMovingAvg_1?6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?)batch_normalization_1/Cast/ReadVariableOp?+batch_normalization_1/Cast_1/ReadVariableOp?%batch_normalization_2/AssignMovingAvg?4batch_normalization_2/AssignMovingAvg/ReadVariableOp?'batch_normalization_2/AssignMovingAvg_1?6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?)batch_normalization_2/Cast/ReadVariableOp?+batch_normalization_2/Cast_1/ReadVariableOp?(city_emb/embedding_lookup/ReadVariableOp?+country_emb/embedding_lookup/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?,statezip_emb/embedding_lookup/ReadVariableOp?*street_emb/embedding_lookup/ReadVariableOp?+weekday_emb/embedding_lookup/ReadVariableOpi
weekday_emb/CastCastinputs_weekday*

DstT0*

SrcT0*'
_output_shapes
:??????????
+weekday_emb/embedding_lookup/ReadVariableOpReadVariableOp4weekday_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype0?
!weekday_emb/embedding_lookup/axisConst*>
_class4
20loc:@weekday_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
weekday_emb/embedding_lookupGatherV23weekday_emb/embedding_lookup/ReadVariableOp:value:0weekday_emb/Cast:y:0*weekday_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@weekday_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:??????????
%weekday_emb/embedding_lookup/IdentityIdentity%weekday_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????i
country_emb/CastCastinputs_country*

DstT0*

SrcT0*'
_output_shapes
:??????????
+country_emb/embedding_lookup/ReadVariableOpReadVariableOp4country_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype0?
!country_emb/embedding_lookup/axisConst*>
_class4
20loc:@country_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
country_emb/embedding_lookupGatherV23country_emb/embedding_lookup/ReadVariableOp:value:0country_emb/Cast:y:0*country_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@country_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:??????????
%country_emb/embedding_lookup/IdentityIdentity%country_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????k
statezip_emb/CastCastinputs_statezip*

DstT0*

SrcT0*'
_output_shapes
:??????????
,statezip_emb/embedding_lookup/ReadVariableOpReadVariableOp5statezip_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:;2*
dtype0?
"statezip_emb/embedding_lookup/axisConst*?
_class5
31loc:@statezip_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
statezip_emb/embedding_lookupGatherV24statezip_emb/embedding_lookup/ReadVariableOp:value:0statezip_emb/Cast:y:0+statezip_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*?
_class5
31loc:@statezip_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????2?
&statezip_emb/embedding_lookup/IdentityIdentity&statezip_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????2c
city_emb/CastCastinputs_city*

DstT0*

SrcT0*'
_output_shapes
:??????????
(city_emb/embedding_lookup/ReadVariableOpReadVariableOp1city_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:-*
dtype0?
city_emb/embedding_lookup/axisConst*;
_class1
/-loc:@city_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
city_emb/embedding_lookupGatherV20city_emb/embedding_lookup/ReadVariableOp:value:0city_emb/Cast:y:0'city_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*;
_class1
/-loc:@city_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????-?
"city_emb/embedding_lookup/IdentityIdentity"city_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????-g
street_emb/CastCastinputs_street*

DstT0*

SrcT0*'
_output_shapes
:??????????
*street_emb/embedding_lookup/ReadVariableOpReadVariableOp3street_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:d*
dtype0?
 street_emb/embedding_lookup/axisConst*=
_class3
1/loc:@street_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
street_emb/embedding_lookupGatherV22street_emb/embedding_lookup/ReadVariableOp:value:0street_emb/Cast:y:0)street_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*=
_class3
1/loc:@street_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????d?
$street_emb/embedding_lookup/IdentityIdentity$street_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????dj
reshape/ShapeShape-street_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshape-street_emb/embedding_lookup/Identity:output:0reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????dj
reshape_1/ShapeShape+city_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :-?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
reshape_1/ReshapeReshape+city_emb/embedding_lookup/Identity:output:0 reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????-n
reshape_2/ShapeShape/statezip_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
reshape_2/ReshapeReshape/statezip_emb/embedding_lookup/Identity:output:0 reshape_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2m
reshape_3/ShapeShape.country_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
reshape_3/ReshapeReshape.country_emb/embedding_lookup/Identity:output:0 reshape_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????m
reshape_4/ShapeShape.weekday_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:g
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
reshape_4/ReshapeReshape.weekday_emb/embedding_lookup/Identity:output:0 reshape_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2inputs_bedroomsinputs_bathroomsinputs_sqft_livinginputs_sqft_lotinputs_floorsinputs_waterfrontinputs_viewinputs_conditioninputs_sqft_aboveinputs_sqft_basementinputs_yr_builtinputs_yr_renovatedreshape/Reshape:output:0reshape_1/Reshape:output:0reshape_2/Reshape:output:0reshape_3/Reshape:output:0reshape_4/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
 batch_normalization/moments/meanMeanconcatenate/concat:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	??
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceconcatenate/concat:output:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:???????????
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 ?
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:??
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:??
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:??
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
#batch_normalization/batchnorm/mul_1Mulconcatenate/concat:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????u
tf.math.tanh/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
"batch_normalization_1/moments/meanMeantf.math.tanh/Tanh:y:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	??
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencetf.math.tanh/Tanh:y:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:???????????
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 ?
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:??
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:??
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0?
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype0?
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:??
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
%batch_normalization_1/batchnorm/mul_1Multf.math.tanh/Tanh:y:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
#batch_normalization_1/batchnorm/subSub1batch_normalization_1/Cast/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
dense/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:????????? ]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? ~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
"batch_normalization_2/moments/meanMeandropout/dropout/Mul_1:z:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(?
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

: ?
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedropout/dropout/Mul_1:z:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:????????? ?
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(?
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 ?
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0?
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
: ?
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: ?
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0?
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
: ?
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ?
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0?
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes
: *
dtype0?
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
: *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: ?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: ?
%batch_normalization_2/batchnorm/mul_1Muldropout/dropout/Mul_1:z:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:????????? ?
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: ?
#batch_normalization_2/batchnorm/subSub1batch_normalization_2/Cast/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_1/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_1/dropout/MulMuldense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????a
dropout_1/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????

NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp)^city_emb/embedding_lookup/ReadVariableOp,^country_emb/embedding_lookup/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp-^statezip_emb/embedding_lookup/ReadVariableOp+^street_emb/embedding_lookup/ReadVariableOp,^weekday_emb/embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2T
(city_emb/embedding_lookup/ReadVariableOp(city_emb/embedding_lookup/ReadVariableOp2Z
+country_emb/embedding_lookup/ReadVariableOp+country_emb/embedding_lookup/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2\
,statezip_emb/embedding_lookup/ReadVariableOp,statezip_emb/embedding_lookup/ReadVariableOp2X
*street_emb/embedding_lookup/ReadVariableOp*street_emb/embedding_lookup/ReadVariableOp2Z
+weekday_emb/embedding_lookup/ReadVariableOp+weekday_emb/embedding_lookup/ReadVariableOp:Y U
'
_output_shapes
:?????????
*
_user_specified_nameinputs/bathrooms:XT
'
_output_shapes
:?????????
)
_user_specified_nameinputs/bedrooms:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/city:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/condition:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/country:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/floors:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/sqft_above:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/sqft_basement:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/sqft_living:X	T
'
_output_shapes
:?????????
)
_user_specified_nameinputs/sqft_lot:X
T
'
_output_shapes
:?????????
)
_user_specified_nameinputs/statezip:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/street:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/view:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/waterfront:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/weekday:XT
'
_output_shapes
:?????????
)
_user_specified_nameinputs/yr_built:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/yr_renovated
?
?
5__inference_batch_normalization_1_layer_call_fn_21574

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19565p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_21788

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_21815

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_reshape_4_layer_call_fn_21426

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_19882`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
a
B__inference_dropout_layer_call_and_return_conditional_losses_21688

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
'__inference_dense_2_layer_call_fn_21824

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_19994o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
^
B__inference_reshape_layer_call_and_return_conditional_losses_19826

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :du
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????dX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_20080

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

+__inference_weekday_emb_layer_call_fn_21343

inputs
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_weekday_emb_layer_call_and_return_conditional_losses_19754s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?*
!__inference__traced_restore_22279
file_prefix8
&assignvariableop_street_emb_embeddings:d8
&assignvariableop_1_city_emb_embeddings:-<
*assignvariableop_2_statezip_emb_embeddings:;2;
)assignvariableop_3_country_emb_embeddings:;
)assignvariableop_4_weekday_emb_embeddings:;
,assignvariableop_5_batch_normalization_gamma:	?:
+assignvariableop_6_batch_normalization_beta:	?A
2assignvariableop_7_batch_normalization_moving_mean:	?E
6assignvariableop_8_batch_normalization_moving_variance:	?=
.assignvariableop_9_batch_normalization_1_gamma:	?=
.assignvariableop_10_batch_normalization_1_beta:	?D
5assignvariableop_11_batch_normalization_1_moving_mean:	?H
9assignvariableop_12_batch_normalization_1_moving_variance:	?3
 assignvariableop_13_dense_kernel:	? ,
assignvariableop_14_dense_bias: =
/assignvariableop_15_batch_normalization_2_gamma: <
.assignvariableop_16_batch_normalization_2_beta: C
5assignvariableop_17_batch_normalization_2_moving_mean: G
9assignvariableop_18_batch_normalization_2_moving_variance: 4
"assignvariableop_19_dense_1_kernel: .
 assignvariableop_20_dense_1_bias:4
"assignvariableop_21_dense_2_kernel:.
 assignvariableop_22_dense_2_bias:'
assignvariableop_23_adam_iter:	 )
assignvariableop_24_adam_beta_1: )
assignvariableop_25_adam_beta_2: (
assignvariableop_26_adam_decay: 0
&assignvariableop_27_adam_learning_rate: %
assignvariableop_28_total_1: %
assignvariableop_29_count_1: #
assignvariableop_30_total: #
assignvariableop_31_count: B
0assignvariableop_32_adam_street_emb_embeddings_m:d@
.assignvariableop_33_adam_city_emb_embeddings_m:-D
2assignvariableop_34_adam_statezip_emb_embeddings_m:;2C
1assignvariableop_35_adam_country_emb_embeddings_m:C
1assignvariableop_36_adam_weekday_emb_embeddings_m:C
4assignvariableop_37_adam_batch_normalization_gamma_m:	?B
3assignvariableop_38_adam_batch_normalization_beta_m:	?E
6assignvariableop_39_adam_batch_normalization_1_gamma_m:	?D
5assignvariableop_40_adam_batch_normalization_1_beta_m:	?:
'assignvariableop_41_adam_dense_kernel_m:	? 3
%assignvariableop_42_adam_dense_bias_m: D
6assignvariableop_43_adam_batch_normalization_2_gamma_m: C
5assignvariableop_44_adam_batch_normalization_2_beta_m: ;
)assignvariableop_45_adam_dense_1_kernel_m: 5
'assignvariableop_46_adam_dense_1_bias_m:;
)assignvariableop_47_adam_dense_2_kernel_m:5
'assignvariableop_48_adam_dense_2_bias_m:B
0assignvariableop_49_adam_street_emb_embeddings_v:d@
.assignvariableop_50_adam_city_emb_embeddings_v:-D
2assignvariableop_51_adam_statezip_emb_embeddings_v:;2C
1assignvariableop_52_adam_country_emb_embeddings_v:C
1assignvariableop_53_adam_weekday_emb_embeddings_v:C
4assignvariableop_54_adam_batch_normalization_gamma_v:	?B
3assignvariableop_55_adam_batch_normalization_beta_v:	?E
6assignvariableop_56_adam_batch_normalization_1_gamma_v:	?D
5assignvariableop_57_adam_batch_normalization_1_beta_v:	?:
'assignvariableop_58_adam_dense_kernel_v:	? 3
%assignvariableop_59_adam_dense_bias_v: D
6assignvariableop_60_adam_batch_normalization_2_gamma_v: C
5assignvariableop_61_adam_batch_normalization_2_beta_v: ;
)assignvariableop_62_adam_dense_1_kernel_v: 5
'assignvariableop_63_adam_dense_1_bias_v:;
)assignvariableop_64_adam_dense_2_kernel_v:5
'assignvariableop_65_adam_dense_2_bias_v:
identity_67??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*?$
value?$B?$CB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*?
value?B?CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Q
dtypesG
E2C	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp&assignvariableop_street_emb_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp&assignvariableop_1_city_emb_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp*assignvariableop_2_statezip_emb_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp)assignvariableop_3_country_emb_embeddingsIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp)assignvariableop_4_weekday_emb_embeddingsIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp,assignvariableop_5_batch_normalization_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp+assignvariableop_6_batch_normalization_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp2assignvariableop_7_batch_normalization_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp6assignvariableop_8_batch_normalization_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_1_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp.assignvariableop_10_batch_normalization_1_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp5assignvariableop_11_batch_normalization_1_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp9assignvariableop_12_batch_normalization_1_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_dense_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_2_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp.assignvariableop_16_batch_normalization_2_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp5assignvariableop_17_batch_normalization_2_moving_meanIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp9assignvariableop_18_batch_normalization_2_moving_varianceIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_dense_1_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_1_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_2_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp assignvariableop_22_dense_2_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_iterIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_decayIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp&assignvariableop_27_adam_learning_rateIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpassignvariableop_28_total_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpassignvariableop_29_count_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpassignvariableop_30_totalIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpassignvariableop_31_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_street_emb_embeddings_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp.assignvariableop_33_adam_city_emb_embeddings_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp2assignvariableop_34_adam_statezip_emb_embeddings_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp1assignvariableop_35_adam_country_emb_embeddings_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp1assignvariableop_36_adam_weekday_emb_embeddings_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_batch_normalization_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_batch_normalization_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_1_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_1_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_dense_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_dense_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_batch_normalization_2_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_batch_normalization_2_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_1_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_1_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_2_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_2_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp0assignvariableop_49_adam_street_emb_embeddings_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp.assignvariableop_50_adam_city_emb_embeddings_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp2assignvariableop_51_adam_statezip_emb_embeddings_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp1assignvariableop_52_adam_country_emb_embeddings_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp1assignvariableop_53_adam_weekday_emb_embeddings_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp4assignvariableop_54_adam_batch_normalization_gamma_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp3assignvariableop_55_adam_batch_normalization_beta_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_batch_normalization_1_gamma_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp5assignvariableop_57_adam_batch_normalization_1_beta_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adam_dense_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp%assignvariableop_59_adam_dense_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_2_gamma_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp5assignvariableop_61_adam_batch_normalization_2_beta_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_dense_1_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp'assignvariableop_63_adam_dense_1_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_dense_2_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp'assignvariableop_65_adam_dense_2_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_66Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_67IdentityIdentity_66:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_67Identity_67:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
F__inference_weekday_emb_layer_call_and_return_conditional_losses_19754

inputs:
(embedding_lookup_readvariableop_resource:
identity??embedding_lookup/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype0?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0Cast:y:0embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:?????????u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:?????????h
NoOpNoOp ^embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
^
B__inference_reshape_layer_call_and_return_conditional_losses_21370

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :du
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????dX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_21798

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_20080o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_street_emb_layer_call_and_return_conditional_losses_21285

inputs:
(embedding_lookup_readvariableop_resource:d
identity??embedding_lookup/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:d*
dtype0?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0Cast:y:0embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????dv
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:?????????du
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:?????????dh
NoOpNoOp ^embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?c
?
@__inference_model_layer_call_and_return_conditional_losses_20001

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16#
weekday_emb_19755:#
country_emb_19769:$
statezip_emb_19783:;2 
city_emb_19797:-"
street_emb_19811:d(
batch_normalization_19908:	?(
batch_normalization_19910:	?(
batch_normalization_19912:	?(
batch_normalization_19914:	?*
batch_normalization_1_19918:	?*
batch_normalization_1_19920:	?*
batch_normalization_1_19922:	?*
batch_normalization_1_19924:	?
dense_19939:	? 
dense_19941: )
batch_normalization_2_19951: )
batch_normalization_2_19953: )
batch_normalization_2_19955: )
batch_normalization_2_19957: 
dense_1_19972: 
dense_1_19974:
dense_2_19995:
dense_2_19997:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall? city_emb/StatefulPartitionedCall?#country_emb/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?$statezip_emb/StatefulPartitionedCall?"street_emb/StatefulPartitionedCall?#weekday_emb/StatefulPartitionedCall?
#weekday_emb/StatefulPartitionedCallStatefulPartitionedCall	inputs_14weekday_emb_19755*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_weekday_emb_layer_call_and_return_conditional_losses_19754?
#country_emb/StatefulPartitionedCallStatefulPartitionedCallinputs_4country_emb_19769*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_country_emb_layer_call_and_return_conditional_losses_19768?
$statezip_emb/StatefulPartitionedCallStatefulPartitionedCall	inputs_10statezip_emb_19783*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_statezip_emb_layer_call_and_return_conditional_losses_19782?
 city_emb/StatefulPartitionedCallStatefulPartitionedCallinputs_2city_emb_19797*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????-*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_city_emb_layer_call_and_return_conditional_losses_19796?
"street_emb/StatefulPartitionedCallStatefulPartitionedCall	inputs_11street_emb_19811*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_street_emb_layer_call_and_return_conditional_losses_19810?
reshape/PartitionedCallPartitionedCall+street_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_19826?
reshape_1/PartitionedCallPartitionedCall)city_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_19840?
reshape_2/PartitionedCallPartitionedCall-statezip_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_19854?
reshape_3/PartitionedCallPartitionedCall,country_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_19868?
reshape_4/PartitionedCallPartitionedCall,weekday_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_19882?
concatenate/PartitionedCallPartitionedCallinputs_1inputsinputs_8inputs_9inputs_5	inputs_13	inputs_12inputs_3inputs_6inputs_7	inputs_15	inputs_16 reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0"reshape_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_19906?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_19908batch_normalization_19910batch_normalization_19912batch_normalization_19914*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19483?
tf.math.tanh/TanhTanh4batch_normalization/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:???????????
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.math.tanh/Tanh:y:0batch_normalization_1_19918batch_normalization_1_19920batch_normalization_1_19922batch_normalization_1_19924*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19565?
dense/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_19939dense_19941*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_19938?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_19949?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0batch_normalization_2_19951batch_normalization_2_19953batch_normalization_2_19955batch_normalization_2_19957*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19647?
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_1_19972dense_1_19974*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_19971?
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_19982?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_19995dense_2_19997*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_19994w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall!^city_emb/StatefulPartitionedCall$^country_emb/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall%^statezip_emb/StatefulPartitionedCall#^street_emb/StatefulPartitionedCall$^weekday_emb/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 city_emb/StatefulPartitionedCall city_emb/StatefulPartitionedCall2J
#country_emb/StatefulPartitionedCall#country_emb/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2L
$statezip_emb/StatefulPartitionedCall$statezip_emb/StatefulPartitionedCall2H
"street_emb/StatefulPartitionedCall"street_emb/StatefulPartitionedCall2J
#weekday_emb/StatefulPartitionedCall#weekday_emb/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_concatenate_layer_call_fn_21459
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_19906a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????d:?????????-:?????????2:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:?????????
#
_user_specified_name	inputs/10:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/11:RN
'
_output_shapes
:?????????d
#
_user_specified_name	inputs/12:RN
'
_output_shapes
:?????????-
#
_user_specified_name	inputs/13:RN
'
_output_shapes
:?????????2
#
_user_specified_name	inputs/14:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/15:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/16
?
?
5__inference_batch_normalization_2_layer_call_fn_21714

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19694o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
`
D__inference_reshape_2_layer_call_and_return_conditional_losses_21404

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_1_layer_call_fn_21587

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19612p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_21803

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21561

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
a
B__inference_dropout_layer_call_and_return_conditional_losses_20113

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?!
?
%__inference_model_layer_call_fn_20819
inputs_bathrooms
inputs_bedrooms
inputs_city
inputs_condition
inputs_country
inputs_floors
inputs_sqft_above
inputs_sqft_basement
inputs_sqft_living
inputs_sqft_lot
inputs_statezip
inputs_street
inputs_view
inputs_waterfront
inputs_weekday
inputs_yr_built
inputs_yr_renovated
unknown:
	unknown_0:
	unknown_1:;2
	unknown_2:-
	unknown_3:d
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	? 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:

unknown_20:

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_bathroomsinputs_bedroomsinputs_cityinputs_conditioninputs_countryinputs_floorsinputs_sqft_aboveinputs_sqft_basementinputs_sqft_livinginputs_sqft_lotinputs_statezipinputs_streetinputs_viewinputs_waterfrontinputs_weekdayinputs_yr_builtinputs_yr_renovatedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*9
_read_only_resource_inputs
 !"#$%&'*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_20001o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_nameinputs/bathrooms:XT
'
_output_shapes
:?????????
)
_user_specified_nameinputs/bedrooms:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/city:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/condition:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/country:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/floors:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/sqft_above:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/sqft_basement:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/sqft_living:X	T
'
_output_shapes
:?????????
)
_user_specified_nameinputs/sqft_lot:X
T
'
_output_shapes
:?????????
)
_user_specified_nameinputs/statezip:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/street:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/view:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/waterfront:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/weekday:XT
'
_output_shapes
:?????????
)
_user_specified_nameinputs/yr_built:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/yr_renovated
?
?
G__inference_statezip_emb_layer_call_and_return_conditional_losses_21319

inputs:
(embedding_lookup_readvariableop_resource:;2
identity??embedding_lookup/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:;2*
dtype0?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0Cast:y:0embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????2v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:?????????2u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:?????????2h
NoOpNoOp ^embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_19840

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :-u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????-X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????-:S O
+
_output_shapes
:?????????-
 
_user_specified_nameinputs
?
|
(__inference_city_emb_layer_call_fn_21292

inputs
unknown:-
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????-*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_city_emb_layer_call_and_return_conditional_losses_19796s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_21834

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_reshape_1_layer_call_fn_21375

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_19840`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????-"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????-:S O
+
_output_shapes
:?????????-
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_21650

inputs
unknown:	? 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_19938o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_reshape_layer_call_fn_21358

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_19826`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
C__inference_city_emb_layer_call_and_return_conditional_losses_19796

inputs:
(embedding_lookup_readvariableop_resource:-
identity??embedding_lookup/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:-*
dtype0?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0Cast:y:0embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????-v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:?????????-u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:?????????-h
NoOpNoOp ^embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19647

inputs*
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: ,
cast_2_readvariableop_resource: ,
cast_3_readvariableop_resource: 
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:????????? k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
E
)__inference_reshape_3_layer_call_fn_21409

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_19868`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
__inference__traced_save_22071
file_prefix4
0savev2_street_emb_embeddings_read_readvariableop2
.savev2_city_emb_embeddings_read_readvariableop6
2savev2_statezip_emb_embeddings_read_readvariableop5
1savev2_country_emb_embeddings_read_readvariableop5
1savev2_weekday_emb_embeddings_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_adam_street_emb_embeddings_m_read_readvariableop9
5savev2_adam_city_emb_embeddings_m_read_readvariableop=
9savev2_adam_statezip_emb_embeddings_m_read_readvariableop<
8savev2_adam_country_emb_embeddings_m_read_readvariableop<
8savev2_adam_weekday_emb_embeddings_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop;
7savev2_adam_street_emb_embeddings_v_read_readvariableop9
5savev2_adam_city_emb_embeddings_v_read_readvariableop=
9savev2_adam_statezip_emb_embeddings_v_read_readvariableop<
8savev2_adam_country_emb_embeddings_v_read_readvariableop<
8savev2_adam_weekday_emb_embeddings_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*?$
value?$B?$CB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*?
value?B?CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_street_emb_embeddings_read_readvariableop.savev2_city_emb_embeddings_read_readvariableop2savev2_statezip_emb_embeddings_read_readvariableop1savev2_country_emb_embeddings_read_readvariableop1savev2_weekday_emb_embeddings_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_adam_street_emb_embeddings_m_read_readvariableop5savev2_adam_city_emb_embeddings_m_read_readvariableop9savev2_adam_statezip_emb_embeddings_m_read_readvariableop8savev2_adam_country_emb_embeddings_m_read_readvariableop8savev2_adam_weekday_emb_embeddings_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop7savev2_adam_street_emb_embeddings_v_read_readvariableop5savev2_adam_city_emb_embeddings_v_read_readvariableop9savev2_adam_statezip_emb_embeddings_v_read_readvariableop8savev2_adam_country_emb_embeddings_v_read_readvariableop8savev2_adam_weekday_emb_embeddings_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Q
dtypesG
E2C	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :d:-:;2:::?:?:?:?:?:?:?:?:	? : : : : : : :::: : : : : : : : : :d:-:;2:::?:?:?:?:	? : : : : ::::d:-:;2:::?:?:?:?:	? : : : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d:$ 

_output_shapes

:-:$ 

_output_shapes

:;2:$ 

_output_shapes

::$ 

_output_shapes

::!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	? : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :$! 

_output_shapes

:d:$" 

_output_shapes

:-:$# 

_output_shapes

:;2:$$ 

_output_shapes

::$% 

_output_shapes

::!&

_output_shapes	
:?:!'

_output_shapes	
:?:!(

_output_shapes	
:?:!)

_output_shapes	
:?:%*!

_output_shapes
:	? : +

_output_shapes
: : ,

_output_shapes
: : -

_output_shapes
: :$. 

_output_shapes

: : /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::$2 

_output_shapes

:d:$3 

_output_shapes

:-:$4 

_output_shapes

:;2:$5 

_output_shapes

::$6 

_output_shapes

::!7

_output_shapes	
:?:!8

_output_shapes	
:?:!9

_output_shapes	
:?:!:

_output_shapes	
:?:%;!

_output_shapes
:	? : <

_output_shapes
: : =

_output_shapes
: : >

_output_shapes
: :$? 

_output_shapes

: : @

_output_shapes
::$A 

_output_shapes

:: B

_output_shapes
::C

_output_shapes
: 
?
?
E__inference_street_emb_layer_call_and_return_conditional_losses_19810

inputs:
(embedding_lookup_readvariableop_resource:d
identity??embedding_lookup/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:d*
dtype0?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0Cast:y:0embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????dv
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:?????????du
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:?????????dh
NoOpNoOp ^embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21641

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?c
?
@__inference_model_layer_call_and_return_conditional_losses_20591
	bathrooms
bedrooms
city
	condition
country

floors

sqft_above
sqft_basement
sqft_living
sqft_lot
statezip

street
view

waterfront
weekday
yr_built
yr_renovated#
weekday_emb_20524:#
country_emb_20527:$
statezip_emb_20530:;2 
city_emb_20533:-"
street_emb_20536:d(
batch_normalization_20545:	?(
batch_normalization_20547:	?(
batch_normalization_20549:	?(
batch_normalization_20551:	?*
batch_normalization_1_20555:	?*
batch_normalization_1_20557:	?*
batch_normalization_1_20559:	?*
batch_normalization_1_20561:	?
dense_20564:	? 
dense_20566: )
batch_normalization_2_20570: )
batch_normalization_2_20572: )
batch_normalization_2_20574: )
batch_normalization_2_20576: 
dense_1_20579: 
dense_1_20581:
dense_2_20585:
dense_2_20587:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall? city_emb/StatefulPartitionedCall?#country_emb/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?$statezip_emb/StatefulPartitionedCall?"street_emb/StatefulPartitionedCall?#weekday_emb/StatefulPartitionedCall?
#weekday_emb/StatefulPartitionedCallStatefulPartitionedCallweekdayweekday_emb_20524*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_weekday_emb_layer_call_and_return_conditional_losses_19754?
#country_emb/StatefulPartitionedCallStatefulPartitionedCallcountrycountry_emb_20527*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_country_emb_layer_call_and_return_conditional_losses_19768?
$statezip_emb/StatefulPartitionedCallStatefulPartitionedCallstatezipstatezip_emb_20530*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_statezip_emb_layer_call_and_return_conditional_losses_19782?
 city_emb/StatefulPartitionedCallStatefulPartitionedCallcitycity_emb_20533*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????-*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_city_emb_layer_call_and_return_conditional_losses_19796?
"street_emb/StatefulPartitionedCallStatefulPartitionedCallstreetstreet_emb_20536*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_street_emb_layer_call_and_return_conditional_losses_19810?
reshape/PartitionedCallPartitionedCall+street_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_19826?
reshape_1/PartitionedCallPartitionedCall)city_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_19840?
reshape_2/PartitionedCallPartitionedCall-statezip_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_19854?
reshape_3/PartitionedCallPartitionedCall,country_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_19868?
reshape_4/PartitionedCallPartitionedCall,weekday_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_19882?
concatenate/PartitionedCallPartitionedCallbedrooms	bathroomssqft_livingsqft_lotfloors
waterfrontview	condition
sqft_abovesqft_basementyr_builtyr_renovated reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0"reshape_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_19906?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_20545batch_normalization_20547batch_normalization_20549batch_normalization_20551*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19483?
tf.math.tanh/TanhTanh4batch_normalization/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:???????????
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.math.tanh/Tanh:y:0batch_normalization_1_20555batch_normalization_1_20557batch_normalization_1_20559batch_normalization_1_20561*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19565?
dense/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_20564dense_20566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_19938?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_19949?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0batch_normalization_2_20570batch_normalization_2_20572batch_normalization_2_20574batch_normalization_2_20576*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19647?
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_1_20579dense_1_20581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_19971?
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_19982?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_20585dense_2_20587*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_19994w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall!^city_emb/StatefulPartitionedCall$^country_emb/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall%^statezip_emb/StatefulPartitionedCall#^street_emb/StatefulPartitionedCall$^weekday_emb/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 city_emb/StatefulPartitionedCall city_emb/StatefulPartitionedCall2J
#country_emb/StatefulPartitionedCall#country_emb/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2L
$statezip_emb/StatefulPartitionedCall$statezip_emb/StatefulPartitionedCall2H
"street_emb/StatefulPartitionedCall"street_emb/StatefulPartitionedCall2J
#weekday_emb/StatefulPartitionedCall#weekday_emb/StatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	bathrooms:QM
'
_output_shapes
:?????????
"
_user_specified_name
bedrooms:MI
'
_output_shapes
:?????????

_user_specified_namecity:RN
'
_output_shapes
:?????????
#
_user_specified_name	condition:PL
'
_output_shapes
:?????????
!
_user_specified_name	country:OK
'
_output_shapes
:?????????
 
_user_specified_namefloors:SO
'
_output_shapes
:?????????
$
_user_specified_name
sqft_above:VR
'
_output_shapes
:?????????
'
_user_specified_namesqft_basement:TP
'
_output_shapes
:?????????
%
_user_specified_namesqft_living:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
sqft_lot:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
statezip:OK
'
_output_shapes
:?????????
 
_user_specified_namestreet:MI
'
_output_shapes
:?????????

_user_specified_nameview:SO
'
_output_shapes
:?????????
$
_user_specified_name
waterfront:PL
'
_output_shapes
:?????????
!
_user_specified_name	weekday:QM
'
_output_shapes
:?????????
"
_user_specified_name
yr_built:UQ
'
_output_shapes
:?????????
&
_user_specified_nameyr_renovated
?
`
'__inference_dropout_layer_call_fn_21671

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_20113o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_21734

inputs*
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: ,
cast_2_readvariableop_resource: ,
cast_3_readvariableop_resource: 
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:????????? k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
`
D__inference_reshape_4_layer_call_and_return_conditional_losses_21438

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_reshape_2_layer_call_fn_21392

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_19854`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????2:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_19982

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19694

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: *
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: ?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:????????? l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: ?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:????????? h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?f
?
@__inference_model_layer_call_and_return_conditional_losses_20677
	bathrooms
bedrooms
city
	condition
country

floors

sqft_above
sqft_basement
sqft_living
sqft_lot
statezip

street
view

waterfront
weekday
yr_built
yr_renovated#
weekday_emb_20610:#
country_emb_20613:$
statezip_emb_20616:;2 
city_emb_20619:-"
street_emb_20622:d(
batch_normalization_20631:	?(
batch_normalization_20633:	?(
batch_normalization_20635:	?(
batch_normalization_20637:	?*
batch_normalization_1_20641:	?*
batch_normalization_1_20643:	?*
batch_normalization_1_20645:	?*
batch_normalization_1_20647:	?
dense_20650:	? 
dense_20652: )
batch_normalization_2_20656: )
batch_normalization_2_20658: )
batch_normalization_2_20660: )
batch_normalization_2_20662: 
dense_1_20665: 
dense_1_20667:
dense_2_20671:
dense_2_20673:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall? city_emb/StatefulPartitionedCall?#country_emb/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?$statezip_emb/StatefulPartitionedCall?"street_emb/StatefulPartitionedCall?#weekday_emb/StatefulPartitionedCall?
#weekday_emb/StatefulPartitionedCallStatefulPartitionedCallweekdayweekday_emb_20610*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_weekday_emb_layer_call_and_return_conditional_losses_19754?
#country_emb/StatefulPartitionedCallStatefulPartitionedCallcountrycountry_emb_20613*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_country_emb_layer_call_and_return_conditional_losses_19768?
$statezip_emb/StatefulPartitionedCallStatefulPartitionedCallstatezipstatezip_emb_20616*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_statezip_emb_layer_call_and_return_conditional_losses_19782?
 city_emb/StatefulPartitionedCallStatefulPartitionedCallcitycity_emb_20619*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????-*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_city_emb_layer_call_and_return_conditional_losses_19796?
"street_emb/StatefulPartitionedCallStatefulPartitionedCallstreetstreet_emb_20622*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_street_emb_layer_call_and_return_conditional_losses_19810?
reshape/PartitionedCallPartitionedCall+street_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_19826?
reshape_1/PartitionedCallPartitionedCall)city_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_19840?
reshape_2/PartitionedCallPartitionedCall-statezip_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_19854?
reshape_3/PartitionedCallPartitionedCall,country_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_19868?
reshape_4/PartitionedCallPartitionedCall,weekday_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_19882?
concatenate/PartitionedCallPartitionedCallbedrooms	bathroomssqft_livingsqft_lotfloors
waterfrontview	condition
sqft_abovesqft_basementyr_builtyr_renovated reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0"reshape_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_19906?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_20631batch_normalization_20633batch_normalization_20635batch_normalization_20637*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19530?
tf.math.tanh/TanhTanh4batch_normalization/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:???????????
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.math.tanh/Tanh:y:0batch_normalization_1_20641batch_normalization_1_20643batch_normalization_1_20645batch_normalization_1_20647*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19612?
dense/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_20650dense_20652*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_19938?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_20113?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0batch_normalization_2_20656batch_normalization_2_20658batch_normalization_2_20660batch_normalization_2_20662*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19694?
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_1_20665dense_1_20667*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_19971?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_20080?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_20671dense_2_20673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_19994w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall!^city_emb/StatefulPartitionedCall$^country_emb/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall%^statezip_emb/StatefulPartitionedCall#^street_emb/StatefulPartitionedCall$^weekday_emb/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 city_emb/StatefulPartitionedCall city_emb/StatefulPartitionedCall2J
#country_emb/StatefulPartitionedCall#country_emb/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2L
$statezip_emb/StatefulPartitionedCall$statezip_emb/StatefulPartitionedCall2H
"street_emb/StatefulPartitionedCall"street_emb/StatefulPartitionedCall2J
#weekday_emb/StatefulPartitionedCall#weekday_emb/StatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	bathrooms:QM
'
_output_shapes
:?????????
"
_user_specified_name
bedrooms:MI
'
_output_shapes
:?????????

_user_specified_namecity:RN
'
_output_shapes
:?????????
#
_user_specified_name	condition:PL
'
_output_shapes
:?????????
!
_user_specified_name	country:OK
'
_output_shapes
:?????????
 
_user_specified_namefloors:SO
'
_output_shapes
:?????????
$
_user_specified_name
sqft_above:VR
'
_output_shapes
:?????????
'
_user_specified_namesqft_basement:TP
'
_output_shapes
:?????????
%
_user_specified_namesqft_living:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
sqft_lot:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
statezip:OK
'
_output_shapes
:?????????
 
_user_specified_namestreet:MI
'
_output_shapes
:?????????

_user_specified_nameview:SO
'
_output_shapes
:?????????
$
_user_specified_name
waterfront:PL
'
_output_shapes
:?????????
!
_user_specified_name	weekday:QM
'
_output_shapes
:?????????
"
_user_specified_name
yr_built:UQ
'
_output_shapes
:?????????
&
_user_specified_nameyr_renovated
?
?
G__inference_statezip_emb_layer_call_and_return_conditional_losses_19782

inputs:
(embedding_lookup_readvariableop_resource:;2
identity??embedding_lookup/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:;2*
dtype0?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0Cast:y:0embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????2v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:?????????2u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:?????????2h
NoOpNoOp ^embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_20752
	bathrooms
bedrooms
city
	condition
country

floors

sqft_above
sqft_basement
sqft_living
sqft_lot
statezip

street
view

waterfront
weekday
yr_built
yr_renovated
unknown:
	unknown_0:
	unknown_1:;2
	unknown_2:-
	unknown_3:d
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	? 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:

unknown_20:

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	bathroomsbedroomscity	conditioncountryfloors
sqft_abovesqft_basementsqft_livingsqft_lotstatezipstreetview
waterfrontweekdayyr_builtyr_renovatedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*9
_read_only_resource_inputs
 !"#$%&'*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_19459o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	bathrooms:QM
'
_output_shapes
:?????????
"
_user_specified_name
bedrooms:MI
'
_output_shapes
:?????????

_user_specified_namecity:RN
'
_output_shapes
:?????????
#
_user_specified_name	condition:PL
'
_output_shapes
:?????????
!
_user_specified_name	country:OK
'
_output_shapes
:?????????
 
_user_specified_namefloors:SO
'
_output_shapes
:?????????
$
_user_specified_name
sqft_above:VR
'
_output_shapes
:?????????
'
_user_specified_namesqft_basement:TP
'
_output_shapes
:?????????
%
_user_specified_namesqft_living:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
sqft_lot:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
statezip:OK
'
_output_shapes
:?????????
 
_user_specified_namestreet:MI
'
_output_shapes
:?????????

_user_specified_nameview:SO
'
_output_shapes
:?????????
$
_user_specified_name
waterfront:PL
'
_output_shapes
:?????????
!
_user_specified_name	weekday:QM
'
_output_shapes
:?????????
"
_user_specified_name
yr_built:UQ
'
_output_shapes
:?????????
&
_user_specified_nameyr_renovated
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21527

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_19994

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_19949

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_21507

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19530p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_concatenate_layer_call_and_return_conditional_losses_19906

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????d:?????????-:?????????2:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????-
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?f
?
@__inference_model_layer_call_and_return_conditional_losses_20389

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16#
weekday_emb_20322:#
country_emb_20325:$
statezip_emb_20328:;2 
city_emb_20331:-"
street_emb_20334:d(
batch_normalization_20343:	?(
batch_normalization_20345:	?(
batch_normalization_20347:	?(
batch_normalization_20349:	?*
batch_normalization_1_20353:	?*
batch_normalization_1_20355:	?*
batch_normalization_1_20357:	?*
batch_normalization_1_20359:	?
dense_20362:	? 
dense_20364: )
batch_normalization_2_20368: )
batch_normalization_2_20370: )
batch_normalization_2_20372: )
batch_normalization_2_20374: 
dense_1_20377: 
dense_1_20379:
dense_2_20383:
dense_2_20385:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall? city_emb/StatefulPartitionedCall?#country_emb/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?$statezip_emb/StatefulPartitionedCall?"street_emb/StatefulPartitionedCall?#weekday_emb/StatefulPartitionedCall?
#weekday_emb/StatefulPartitionedCallStatefulPartitionedCall	inputs_14weekday_emb_20322*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_weekday_emb_layer_call_and_return_conditional_losses_19754?
#country_emb/StatefulPartitionedCallStatefulPartitionedCallinputs_4country_emb_20325*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_country_emb_layer_call_and_return_conditional_losses_19768?
$statezip_emb/StatefulPartitionedCallStatefulPartitionedCall	inputs_10statezip_emb_20328*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_statezip_emb_layer_call_and_return_conditional_losses_19782?
 city_emb/StatefulPartitionedCallStatefulPartitionedCallinputs_2city_emb_20331*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????-*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_city_emb_layer_call_and_return_conditional_losses_19796?
"street_emb/StatefulPartitionedCallStatefulPartitionedCall	inputs_11street_emb_20334*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_street_emb_layer_call_and_return_conditional_losses_19810?
reshape/PartitionedCallPartitionedCall+street_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_19826?
reshape_1/PartitionedCallPartitionedCall)city_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_19840?
reshape_2/PartitionedCallPartitionedCall-statezip_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_19854?
reshape_3/PartitionedCallPartitionedCall,country_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_19868?
reshape_4/PartitionedCallPartitionedCall,weekday_emb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_19882?
concatenate/PartitionedCallPartitionedCallinputs_1inputsinputs_8inputs_9inputs_5	inputs_13	inputs_12inputs_3inputs_6inputs_7	inputs_15	inputs_16 reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0"reshape_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_19906?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_20343batch_normalization_20345batch_normalization_20347batch_normalization_20349*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19530?
tf.math.tanh/TanhTanh4batch_normalization/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:???????????
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCalltf.math.tanh/Tanh:y:0batch_normalization_1_20353batch_normalization_1_20355batch_normalization_1_20357batch_normalization_1_20359*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19612?
dense/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_20362dense_20364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_19938?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_20113?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0batch_normalization_2_20368batch_normalization_2_20370batch_normalization_2_20372batch_normalization_2_20374*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19694?
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_1_20377dense_1_20379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_19971?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_20080?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_20383dense_2_20385*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_19994w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall!^city_emb/StatefulPartitionedCall$^country_emb/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall%^statezip_emb/StatefulPartitionedCall#^street_emb/StatefulPartitionedCall$^weekday_emb/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 city_emb/StatefulPartitionedCall city_emb/StatefulPartitionedCall2J
#country_emb/StatefulPartitionedCall#country_emb/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2L
$statezip_emb/StatefulPartitionedCall$statezip_emb/StatefulPartitionedCall2H
"street_emb/StatefulPartitionedCall"street_emb/StatefulPartitionedCall2J
#weekday_emb/StatefulPartitionedCall#weekday_emb/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_19612

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_19459
	bathrooms
bedrooms
city
	condition
country

floors

sqft_above
sqft_basement
sqft_living
sqft_lot
statezip

street
view

waterfront
weekday
yr_built
yr_renovatedL
:model_weekday_emb_embedding_lookup_readvariableop_resource:L
:model_country_emb_embedding_lookup_readvariableop_resource:M
;model_statezip_emb_embedding_lookup_readvariableop_resource:;2I
7model_city_emb_embedding_lookup_readvariableop_resource:-K
9model_street_emb_embedding_lookup_readvariableop_resource:dE
6model_batch_normalization_cast_readvariableop_resource:	?G
8model_batch_normalization_cast_1_readvariableop_resource:	?G
8model_batch_normalization_cast_2_readvariableop_resource:	?G
8model_batch_normalization_cast_3_readvariableop_resource:	?G
8model_batch_normalization_1_cast_readvariableop_resource:	?I
:model_batch_normalization_1_cast_1_readvariableop_resource:	?I
:model_batch_normalization_1_cast_2_readvariableop_resource:	?I
:model_batch_normalization_1_cast_3_readvariableop_resource:	?=
*model_dense_matmul_readvariableop_resource:	? 9
+model_dense_biasadd_readvariableop_resource: F
8model_batch_normalization_2_cast_readvariableop_resource: H
:model_batch_normalization_2_cast_1_readvariableop_resource: H
:model_batch_normalization_2_cast_2_readvariableop_resource: H
:model_batch_normalization_2_cast_3_readvariableop_resource: >
,model_dense_1_matmul_readvariableop_resource: ;
-model_dense_1_biasadd_readvariableop_resource:>
,model_dense_2_matmul_readvariableop_resource:;
-model_dense_2_biasadd_readvariableop_resource:
identity??-model/batch_normalization/Cast/ReadVariableOp?/model/batch_normalization/Cast_1/ReadVariableOp?/model/batch_normalization/Cast_2/ReadVariableOp?/model/batch_normalization/Cast_3/ReadVariableOp?/model/batch_normalization_1/Cast/ReadVariableOp?1model/batch_normalization_1/Cast_1/ReadVariableOp?1model/batch_normalization_1/Cast_2/ReadVariableOp?1model/batch_normalization_1/Cast_3/ReadVariableOp?/model/batch_normalization_2/Cast/ReadVariableOp?1model/batch_normalization_2/Cast_1/ReadVariableOp?1model/batch_normalization_2/Cast_2/ReadVariableOp?1model/batch_normalization_2/Cast_3/ReadVariableOp?.model/city_emb/embedding_lookup/ReadVariableOp?1model/country_emb/embedding_lookup/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?$model/dense_2/BiasAdd/ReadVariableOp?#model/dense_2/MatMul/ReadVariableOp?2model/statezip_emb/embedding_lookup/ReadVariableOp?0model/street_emb/embedding_lookup/ReadVariableOp?1model/weekday_emb/embedding_lookup/ReadVariableOph
model/weekday_emb/CastCastweekday*

DstT0*

SrcT0*'
_output_shapes
:??????????
1model/weekday_emb/embedding_lookup/ReadVariableOpReadVariableOp:model_weekday_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype0?
'model/weekday_emb/embedding_lookup/axisConst*D
_class:
86loc:@model/weekday_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
"model/weekday_emb/embedding_lookupGatherV29model/weekday_emb/embedding_lookup/ReadVariableOp:value:0model/weekday_emb/Cast:y:00model/weekday_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*D
_class:
86loc:@model/weekday_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:??????????
+model/weekday_emb/embedding_lookup/IdentityIdentity+model/weekday_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????h
model/country_emb/CastCastcountry*

DstT0*

SrcT0*'
_output_shapes
:??????????
1model/country_emb/embedding_lookup/ReadVariableOpReadVariableOp:model_country_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype0?
'model/country_emb/embedding_lookup/axisConst*D
_class:
86loc:@model/country_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
"model/country_emb/embedding_lookupGatherV29model/country_emb/embedding_lookup/ReadVariableOp:value:0model/country_emb/Cast:y:00model/country_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*D
_class:
86loc:@model/country_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:??????????
+model/country_emb/embedding_lookup/IdentityIdentity+model/country_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????j
model/statezip_emb/CastCaststatezip*

DstT0*

SrcT0*'
_output_shapes
:??????????
2model/statezip_emb/embedding_lookup/ReadVariableOpReadVariableOp;model_statezip_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:;2*
dtype0?
(model/statezip_emb/embedding_lookup/axisConst*E
_class;
97loc:@model/statezip_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
#model/statezip_emb/embedding_lookupGatherV2:model/statezip_emb/embedding_lookup/ReadVariableOp:value:0model/statezip_emb/Cast:y:01model/statezip_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*E
_class;
97loc:@model/statezip_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????2?
,model/statezip_emb/embedding_lookup/IdentityIdentity,model/statezip_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????2b
model/city_emb/CastCastcity*

DstT0*

SrcT0*'
_output_shapes
:??????????
.model/city_emb/embedding_lookup/ReadVariableOpReadVariableOp7model_city_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:-*
dtype0?
$model/city_emb/embedding_lookup/axisConst*A
_class7
53loc:@model/city_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
model/city_emb/embedding_lookupGatherV26model/city_emb/embedding_lookup/ReadVariableOp:value:0model/city_emb/Cast:y:0-model/city_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*A
_class7
53loc:@model/city_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????-?
(model/city_emb/embedding_lookup/IdentityIdentity(model/city_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????-f
model/street_emb/CastCaststreet*

DstT0*

SrcT0*'
_output_shapes
:??????????
0model/street_emb/embedding_lookup/ReadVariableOpReadVariableOp9model_street_emb_embedding_lookup_readvariableop_resource*
_output_shapes

:d*
dtype0?
&model/street_emb/embedding_lookup/axisConst*C
_class9
75loc:@model/street_emb/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
!model/street_emb/embedding_lookupGatherV28model/street_emb/embedding_lookup/ReadVariableOp:value:0model/street_emb/Cast:y:0/model/street_emb/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*C
_class9
75loc:@model/street_emb/embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????d?
*model/street_emb/embedding_lookup/IdentityIdentity*model/street_emb/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????dv
model/reshape/ShapeShape3model/street_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:k
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d?
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/reshape/ReshapeReshape3model/street_emb/embedding_lookup/Identity:output:0$model/reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????dv
model/reshape_1/ShapeShape1model/city_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:m
#model/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model/reshape_1/strided_sliceStridedSlicemodel/reshape_1/Shape:output:0,model/reshape_1/strided_slice/stack:output:0.model/reshape_1/strided_slice/stack_1:output:0.model/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :-?
model/reshape_1/Reshape/shapePack&model/reshape_1/strided_slice:output:0(model/reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/reshape_1/ReshapeReshape1model/city_emb/embedding_lookup/Identity:output:0&model/reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????-z
model/reshape_2/ShapeShape5model/statezip_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:m
#model/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model/reshape_2/strided_sliceStridedSlicemodel/reshape_2/Shape:output:0,model/reshape_2/strided_slice/stack:output:0.model/reshape_2/strided_slice/stack_1:output:0.model/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2?
model/reshape_2/Reshape/shapePack&model/reshape_2/strided_slice:output:0(model/reshape_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/reshape_2/ReshapeReshape5model/statezip_emb/embedding_lookup/Identity:output:0&model/reshape_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2y
model/reshape_3/ShapeShape4model/country_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:m
#model/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model/reshape_3/strided_sliceStridedSlicemodel/reshape_3/Shape:output:0,model/reshape_3/strided_slice/stack:output:0.model/reshape_3/strided_slice/stack_1:output:0.model/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
model/reshape_3/Reshape/shapePack&model/reshape_3/strided_slice:output:0(model/reshape_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/reshape_3/ReshapeReshape4model/country_emb/embedding_lookup/Identity:output:0&model/reshape_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????y
model/reshape_4/ShapeShape4model/weekday_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:m
#model/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model/reshape_4/strided_sliceStridedSlicemodel/reshape_4/Shape:output:0,model/reshape_4/strided_slice/stack:output:0.model/reshape_4/strided_slice/stack_1:output:0.model/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
model/reshape_4/Reshape/shapePack&model/reshape_4/strided_slice:output:0(model/reshape_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/reshape_4/ReshapeReshape4model/weekday_emb/embedding_lookup/Identity:output:0&model/reshape_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model/concatenate/concatConcatV2bedrooms	bathroomssqft_livingsqft_lotfloors
waterfrontview	condition
sqft_abovesqft_basementyr_builtyr_renovatedmodel/reshape/Reshape:output:0 model/reshape_1/Reshape:output:0 model/reshape_2/Reshape:output:0 model/reshape_3/Reshape:output:0 model/reshape_4/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
-model/batch_normalization/Cast/ReadVariableOpReadVariableOp6model_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/model/batch_normalization/Cast_1/ReadVariableOpReadVariableOp8model_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/model/batch_normalization/Cast_2/ReadVariableOpReadVariableOp8model_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/model/batch_normalization/Cast_3/ReadVariableOpReadVariableOp8model_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
'model/batch_normalization/batchnorm/addAddV27model/batch_normalization/Cast_1/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:??
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:??
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:07model/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
)model/batch_normalization/batchnorm/mul_1Mul!model/concatenate/concat:output:0+model/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
)model/batch_normalization/batchnorm/mul_2Mul5model/batch_normalization/Cast/ReadVariableOp:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
'model/batch_normalization/batchnorm/subSub7model/batch_normalization/Cast_2/ReadVariableOp:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
model/tf.math.tanh/TanhTanh-model/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:???????????
/model/batch_normalization_1/Cast/ReadVariableOpReadVariableOp8model_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1model/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp:model_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1model/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp:model_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1model/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp:model_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype0p
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
)model/batch_normalization_1/batchnorm/addAddV29model/batch_normalization_1/Cast_1/ReadVariableOp:value:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:??
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:??
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:09model/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
+model/batch_normalization_1/batchnorm/mul_1Mulmodel/tf.math.tanh/Tanh:y:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
+model/batch_normalization_1/batchnorm/mul_2Mul7model/batch_normalization_1/Cast/ReadVariableOp:value:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
)model/batch_normalization_1/batchnorm/subSub9model/batch_normalization_1/Cast_2/ReadVariableOp:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
model/dense/MatMulMatMul/model/batch_normalization_1/batchnorm/add_1:z:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? t
model/dropout/IdentityIdentitymodel/dense/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
/model/batch_normalization_2/Cast/ReadVariableOpReadVariableOp8model_batch_normalization_2_cast_readvariableop_resource*
_output_shapes
: *
dtype0?
1model/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp:model_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
: *
dtype0?
1model/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp:model_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0?
1model/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp:model_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes
: *
dtype0p
+model/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
)model/batch_normalization_2/batchnorm/addAddV29model/batch_normalization_2/Cast_1/ReadVariableOp:value:04model/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: ?
+model/batch_normalization_2/batchnorm/RsqrtRsqrt-model/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: ?
)model/batch_normalization_2/batchnorm/mulMul/model/batch_normalization_2/batchnorm/Rsqrt:y:09model/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: ?
+model/batch_normalization_2/batchnorm/mul_1Mulmodel/dropout/Identity:output:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:????????? ?
+model/batch_normalization_2/batchnorm/mul_2Mul7model/batch_normalization_2/Cast/ReadVariableOp:value:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: ?
)model/batch_normalization_2/batchnorm/subSub9model/batch_normalization_2/Cast_2/ReadVariableOp:value:0/model/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ?
+model/batch_normalization_2/batchnorm/add_1AddV2/model/batch_normalization_2/batchnorm/mul_1:z:0-model/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? ?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
model/dense_1/MatMulMatMul/model/batch_normalization_2/batchnorm/add_1:z:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????l
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x
model/dropout_1/IdentityIdentity model/dense_1/Relu:activations:0*
T0*'
_output_shapes
:??????????
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model/dense_2/MatMulMatMul!model/dropout_1/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitymodel/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp.^model/batch_normalization/Cast/ReadVariableOp0^model/batch_normalization/Cast_1/ReadVariableOp0^model/batch_normalization/Cast_2/ReadVariableOp0^model/batch_normalization/Cast_3/ReadVariableOp0^model/batch_normalization_1/Cast/ReadVariableOp2^model/batch_normalization_1/Cast_1/ReadVariableOp2^model/batch_normalization_1/Cast_2/ReadVariableOp2^model/batch_normalization_1/Cast_3/ReadVariableOp0^model/batch_normalization_2/Cast/ReadVariableOp2^model/batch_normalization_2/Cast_1/ReadVariableOp2^model/batch_normalization_2/Cast_2/ReadVariableOp2^model/batch_normalization_2/Cast_3/ReadVariableOp/^model/city_emb/embedding_lookup/ReadVariableOp2^model/country_emb/embedding_lookup/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp3^model/statezip_emb/embedding_lookup/ReadVariableOp1^model/street_emb/embedding_lookup/ReadVariableOp2^model/weekday_emb/embedding_lookup/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : 2^
-model/batch_normalization/Cast/ReadVariableOp-model/batch_normalization/Cast/ReadVariableOp2b
/model/batch_normalization/Cast_1/ReadVariableOp/model/batch_normalization/Cast_1/ReadVariableOp2b
/model/batch_normalization/Cast_2/ReadVariableOp/model/batch_normalization/Cast_2/ReadVariableOp2b
/model/batch_normalization/Cast_3/ReadVariableOp/model/batch_normalization/Cast_3/ReadVariableOp2b
/model/batch_normalization_1/Cast/ReadVariableOp/model/batch_normalization_1/Cast/ReadVariableOp2f
1model/batch_normalization_1/Cast_1/ReadVariableOp1model/batch_normalization_1/Cast_1/ReadVariableOp2f
1model/batch_normalization_1/Cast_2/ReadVariableOp1model/batch_normalization_1/Cast_2/ReadVariableOp2f
1model/batch_normalization_1/Cast_3/ReadVariableOp1model/batch_normalization_1/Cast_3/ReadVariableOp2b
/model/batch_normalization_2/Cast/ReadVariableOp/model/batch_normalization_2/Cast/ReadVariableOp2f
1model/batch_normalization_2/Cast_1/ReadVariableOp1model/batch_normalization_2/Cast_1/ReadVariableOp2f
1model/batch_normalization_2/Cast_2/ReadVariableOp1model/batch_normalization_2/Cast_2/ReadVariableOp2f
1model/batch_normalization_2/Cast_3/ReadVariableOp1model/batch_normalization_2/Cast_3/ReadVariableOp2`
.model/city_emb/embedding_lookup/ReadVariableOp.model/city_emb/embedding_lookup/ReadVariableOp2f
1model/country_emb/embedding_lookup/ReadVariableOp1model/country_emb/embedding_lookup/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2h
2model/statezip_emb/embedding_lookup/ReadVariableOp2model/statezip_emb/embedding_lookup/ReadVariableOp2d
0model/street_emb/embedding_lookup/ReadVariableOp0model/street_emb/embedding_lookup/ReadVariableOp2f
1model/weekday_emb/embedding_lookup/ReadVariableOp1model/weekday_emb/embedding_lookup/ReadVariableOp:R N
'
_output_shapes
:?????????
#
_user_specified_name	bathrooms:QM
'
_output_shapes
:?????????
"
_user_specified_name
bedrooms:MI
'
_output_shapes
:?????????

_user_specified_namecity:RN
'
_output_shapes
:?????????
#
_user_specified_name	condition:PL
'
_output_shapes
:?????????
!
_user_specified_name	country:OK
'
_output_shapes
:?????????
 
_user_specified_namefloors:SO
'
_output_shapes
:?????????
$
_user_specified_name
sqft_above:VR
'
_output_shapes
:?????????
'
_user_specified_namesqft_basement:TP
'
_output_shapes
:?????????
%
_user_specified_namesqft_living:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
sqft_lot:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
statezip:OK
'
_output_shapes
:?????????
 
_user_specified_namestreet:MI
'
_output_shapes
:?????????

_user_specified_nameview:SO
'
_output_shapes
:?????????
$
_user_specified_name
waterfront:PL
'
_output_shapes
:?????????
!
_user_specified_name	weekday:QM
'
_output_shapes
:?????????
"
_user_specified_name
yr_built:UQ
'
_output_shapes
:?????????
&
_user_specified_nameyr_renovated
?
?
'__inference_dense_1_layer_call_fn_21777

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_19971o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?	
serving_default?	
?
	bathrooms2
serving_default_bathrooms:0?????????
=
bedrooms1
serving_default_bedrooms:0?????????
5
city-
serving_default_city:0?????????
?
	condition2
serving_default_condition:0?????????
;
country0
serving_default_country:0?????????
9
floors/
serving_default_floors:0?????????
A

sqft_above3
serving_default_sqft_above:0?????????
G
sqft_basement6
serving_default_sqft_basement:0?????????
C
sqft_living4
serving_default_sqft_living:0?????????
=
sqft_lot1
serving_default_sqft_lot:0?????????
=
statezip1
serving_default_statezip:0?????????
9
street/
serving_default_street:0?????????
5
view-
serving_default_view:0?????????
A

waterfront3
serving_default_waterfront:0?????????
;
weekday0
serving_default_weekday:0?????????
=
yr_built1
serving_default_yr_built:0?????????
E
yr_renovated5
serving_default_yr_renovated:0?????????;
dense_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer_with_weights-5
layer-28
layer-29
layer_with_weights-6
layer-30
 layer_with_weights-7
 layer-31
!layer-32
"layer_with_weights-8
"layer-33
#layer_with_weights-9
#layer-34
$layer-35
%layer_with_weights-10
%layer-36
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,_default_save_signature
-	optimizer
.
signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5
embeddings"
_tf_keras_layer
?
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<
embeddings"
_tf_keras_layer
?
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
C
embeddings"
_tf_keras_layer
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
J
embeddings"
_tf_keras_layer
?
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses
Q
embeddings"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
?
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
?
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
?
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
?
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|axis
	}gamma
~beta
moving_mean
?moving_variance"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
50
<1
C2
J3
Q4
}5
~6
7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22"
trackable_list_wrapper
?
50
<1
C2
J3
Q4
}5
~6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
,_default_save_signature
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
%__inference_model_layer_call_fn_20050
%__inference_model_layer_call_fn_20819
%__inference_model_layer_call_fn_20886
%__inference_model_layer_call_fn_20505?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
@__inference_model_layer_call_and_return_conditional_losses_21049
@__inference_model_layer_call_and_return_conditional_losses_21268
@__inference_model_layer_call_and_return_conditional_losses_20591
@__inference_model_layer_call_and_return_conditional_losses_20677?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?B?
 __inference__wrapped_model_19459	bathroomsbedroomscity	conditioncountryfloors
sqft_abovesqft_basementsqft_livingsqft_lotstatezipstreetview
waterfrontweekdayyr_builtyr_renovated"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate5m?<m?Cm?Jm?Qm?}m?~m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?5v?<v?Cv?Jv?Qv?}v?~v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
-
?serving_default"
signature_map
'
50"
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_street_emb_layer_call_fn_21275?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_street_emb_layer_call_and_return_conditional_losses_21285?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
):'d 2street_emb/embeddings
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_city_emb_layer_call_fn_21292?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_city_emb_layer_call_and_return_conditional_losses_21302?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
':%- 2city_emb/embeddings
'
C0"
trackable_list_wrapper
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
,__inference_statezip_emb_layer_call_fn_21309?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
G__inference_statezip_emb_layer_call_and_return_conditional_losses_21319?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
+:);2 2statezip_emb/embeddings
'
J0"
trackable_list_wrapper
'
J0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_country_emb_layer_call_fn_21326?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_country_emb_layer_call_and_return_conditional_losses_21336?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:( 2country_emb/embeddings
'
Q0"
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_weekday_emb_layer_call_fn_21343?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_weekday_emb_layer_call_and_return_conditional_losses_21353?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:( 2weekday_emb/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_reshape_layer_call_fn_21358?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_reshape_layer_call_and_return_conditional_losses_21370?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_reshape_1_layer_call_fn_21375?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_reshape_1_layer_call_and_return_conditional_losses_21387?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_reshape_2_layer_call_fn_21392?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_reshape_2_layer_call_and_return_conditional_losses_21404?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_reshape_3_layer_call_fn_21409?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_reshape_3_layer_call_and_return_conditional_losses_21421?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_reshape_4_layer_call_fn_21426?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_reshape_4_layer_call_and_return_conditional_losses_21438?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_concatenate_layer_call_fn_21459?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_concatenate_layer_call_and_return_conditional_losses_21481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
=
}0
~1
2
?3"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
3__inference_batch_normalization_layer_call_fn_21494
3__inference_batch_normalization_layer_call_fn_21507?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21527
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21561?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:(? 2batch_normalization/gamma
):'? 2batch_normalization/beta
0:.? (2batch_normalization/moving_mean
4:2? (2#batch_normalization/moving_variance
"
_generic_user_object
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
5__inference_batch_normalization_1_layer_call_fn_21574
5__inference_batch_normalization_1_layer_call_fn_21587?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21607
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21641?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
,:*? 2batch_normalization_1/gamma
+:)? 2batch_normalization_1/beta
2:0? (2!batch_normalization_1/moving_mean
6:4? (2%batch_normalization_1/moving_variance
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
%__inference_dense_layer_call_fn_21650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
@__inference_dense_layer_call_and_return_conditional_losses_21661?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:	?  2dense/kernel
:  2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
'__inference_dropout_layer_call_fn_21666
'__inference_dropout_layer_call_fn_21671?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
B__inference_dropout_layer_call_and_return_conditional_losses_21676
B__inference_dropout_layer_call_and_return_conditional_losses_21688?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
5__inference_batch_normalization_2_layer_call_fn_21701
5__inference_batch_normalization_2_layer_call_fn_21714?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_21734
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_21768?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
+:)  2batch_normalization_2/gamma
*:(  2batch_normalization_2/beta
1:/  (2!batch_normalization_2/moving_mean
5:3  (2%batch_normalization_2/moving_variance
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_1_layer_call_fn_21777?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_1_layer_call_and_return_conditional_losses_21788?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
":   2dense_1/kernel
: 2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_1_layer_call_fn_21793
)__inference_dropout_1_layer_call_fn_21798?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_1_layer_call_and_return_conditional_losses_21803
D__inference_dropout_1_layer_call_and_return_conditional_losses_21815?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_2_layer_call_fn_21824?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_2_layer_call_and_return_conditional_losses_21834?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
":  2dense_2/kernel
: 2dense_2/bias
O
0
?1
?2
?3
?4
?5"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_model_layer_call_fn_20050	bathroomsbedroomscity	conditioncountryfloors
sqft_abovesqft_basementsqft_livingsqft_lotstatezipstreetview
waterfrontweekdayyr_builtyr_renovated"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_20819inputs/bathroomsinputs/bedroomsinputs/cityinputs/conditioninputs/countryinputs/floorsinputs/sqft_aboveinputs/sqft_basementinputs/sqft_livinginputs/sqft_lotinputs/statezipinputs/streetinputs/viewinputs/waterfrontinputs/weekdayinputs/yr_builtinputs/yr_renovated"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_20886inputs/bathroomsinputs/bedroomsinputs/cityinputs/conditioninputs/countryinputs/floorsinputs/sqft_aboveinputs/sqft_basementinputs/sqft_livinginputs/sqft_lotinputs/statezipinputs/streetinputs/viewinputs/waterfrontinputs/weekdayinputs/yr_builtinputs/yr_renovated"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_20505	bathroomsbedroomscity	conditioncountryfloors
sqft_abovesqft_basementsqft_livingsqft_lotstatezipstreetview
waterfrontweekdayyr_builtyr_renovated"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_21049inputs/bathroomsinputs/bedroomsinputs/cityinputs/conditioninputs/countryinputs/floorsinputs/sqft_aboveinputs/sqft_basementinputs/sqft_livinginputs/sqft_lotinputs/statezipinputs/streetinputs/viewinputs/waterfrontinputs/weekdayinputs/yr_builtinputs/yr_renovated"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_21268inputs/bathroomsinputs/bedroomsinputs/cityinputs/conditioninputs/countryinputs/floorsinputs/sqft_aboveinputs/sqft_basementinputs/sqft_livinginputs/sqft_lotinputs/statezipinputs/streetinputs/viewinputs/waterfrontinputs/weekdayinputs/yr_builtinputs/yr_renovated"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_20591	bathroomsbedroomscity	conditioncountryfloors
sqft_abovesqft_basementsqft_livingsqft_lotstatezipstreetview
waterfrontweekdayyr_builtyr_renovated"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_20677	bathroomsbedroomscity	conditioncountryfloors
sqft_abovesqft_basementsqft_livingsqft_lotstatezipstreetview
waterfrontweekdayyr_builtyr_renovated"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	  (2	Adam/iter
:  (2Adam/beta_1
:  (2Adam/beta_2
:  (2
Adam/decay
:  (2Adam/learning_rate
?B?
#__inference_signature_wrapper_20752	bathroomsbedroomscity	conditioncountryfloors
sqft_abovesqft_basementsqft_livingsqft_lotstatezipstreetview
waterfrontweekdayyr_builtyr_renovated"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
*__inference_street_emb_layer_call_fn_21275inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_street_emb_layer_call_and_return_conditional_losses_21285inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
(__inference_city_emb_layer_call_fn_21292inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_city_emb_layer_call_and_return_conditional_losses_21302inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
,__inference_statezip_emb_layer_call_fn_21309inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_statezip_emb_layer_call_and_return_conditional_losses_21319inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
+__inference_country_emb_layer_call_fn_21326inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_country_emb_layer_call_and_return_conditional_losses_21336inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
+__inference_weekday_emb_layer_call_fn_21343inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_weekday_emb_layer_call_and_return_conditional_losses_21353inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
'__inference_reshape_layer_call_fn_21358inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_reshape_layer_call_and_return_conditional_losses_21370inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_reshape_1_layer_call_fn_21375inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_reshape_1_layer_call_and_return_conditional_losses_21387inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_reshape_2_layer_call_fn_21392inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_reshape_2_layer_call_and_return_conditional_losses_21404inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_reshape_3_layer_call_fn_21409inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_reshape_3_layer_call_and_return_conditional_losses_21421inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_reshape_4_layer_call_fn_21426inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_reshape_4_layer_call_and_return_conditional_losses_21438inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
+__inference_concatenate_layer_call_fn_21459inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10	inputs/11	inputs/12	inputs/13	inputs/14	inputs/15	inputs/16"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_concatenate_layer_call_and_return_conditional_losses_21481inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10	inputs/11	inputs/12	inputs/13	inputs/14	inputs/15	inputs/16"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
/
0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
3__inference_batch_normalization_layer_call_fn_21494inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
3__inference_batch_normalization_layer_call_fn_21507inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21527inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21561inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_batch_normalization_1_layer_call_fn_21574inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
5__inference_batch_normalization_1_layer_call_fn_21587inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21607inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21641inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
%__inference_dense_layer_call_fn_21650inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
@__inference_dense_layer_call_and_return_conditional_losses_21661inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
'__inference_dropout_layer_call_fn_21666inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_dropout_layer_call_fn_21671inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_21676inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_21688inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_batch_normalization_2_layer_call_fn_21701inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
5__inference_batch_normalization_2_layer_call_fn_21714inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_21734inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_21768inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
'__inference_dense_1_layer_call_fn_21777inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_1_layer_call_and_return_conditional_losses_21788inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_dropout_1_layer_call_fn_21793inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
)__inference_dropout_1_layer_call_fn_21798inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_1_layer_call_and_return_conditional_losses_21803inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_1_layer_call_and_return_conditional_losses_21815inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
'__inference_dense_2_layer_call_fn_21824inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_2_layer_call_and_return_conditional_losses_21834inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.:,d 2Adam/street_emb/embeddings/m
,:*- 2Adam/city_emb/embeddings/m
0:.;2 2Adam/statezip_emb/embeddings/m
/:- 2Adam/country_emb/embeddings/m
/:- 2Adam/weekday_emb/embeddings/m
/:-? 2 Adam/batch_normalization/gamma/m
.:,? 2Adam/batch_normalization/beta/m
1:/? 2"Adam/batch_normalization_1/gamma/m
0:.? 2!Adam/batch_normalization_1/beta/m
&:$	?  2Adam/dense/kernel/m
:  2Adam/dense/bias/m
0:.  2"Adam/batch_normalization_2/gamma/m
/:-  2!Adam/batch_normalization_2/beta/m
':%  2Adam/dense_1/kernel/m
!: 2Adam/dense_1/bias/m
':% 2Adam/dense_2/kernel/m
!: 2Adam/dense_2/bias/m
.:,d 2Adam/street_emb/embeddings/v
,:*- 2Adam/city_emb/embeddings/v
0:.;2 2Adam/statezip_emb/embeddings/v
/:- 2Adam/country_emb/embeddings/v
/:- 2Adam/weekday_emb/embeddings/v
/:-? 2 Adam/batch_normalization/gamma/v
.:,? 2Adam/batch_normalization/beta/v
1:/? 2"Adam/batch_normalization_1/gamma/v
0:.? 2!Adam/batch_normalization_1/beta/v
&:$	?  2Adam/dense/kernel/v
:  2Adam/dense/bias/v
0:.  2"Adam/batch_normalization_2/gamma/v
/:-  2!Adam/batch_normalization_2/beta/v
':%  2Adam/dense_1/kernel/v
!: 2Adam/dense_1/bias/v
':% 2Adam/dense_2/kernel/v
!: 2Adam/dense_2/bias/v?
 __inference__wrapped_model_19459?&QJC<5?~}?????????????????
???
???
0
	bathrooms#? 
	bathrooms?????????
.
bedrooms"?
bedrooms?????????
&
city?
city?????????
0
	condition#? 
	condition?????????
,
country!?
country?????????
*
floors ?
floors?????????
2

sqft_above$?!

sqft_above?????????
8
sqft_basement'?$
sqft_basement?????????
4
sqft_living%?"
sqft_living?????????
.
sqft_lot"?
sqft_lot?????????
.
statezip"?
statezip?????????
*
street ?
street?????????
&
view?
view?????????
2

waterfront$?!

waterfront?????????
,
weekday!?
weekday?????????
.
yr_built"?
yr_built?????????
6
yr_renovated&?#
yr_renovated?????????
? "1?.
,
dense_2!?
dense_2??????????
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21607h????4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21641h????4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
5__inference_batch_normalization_1_layer_call_fn_21574[????4?1
*?'
!?
inputs??????????
p 
? "????????????
5__inference_batch_normalization_1_layer_call_fn_21587[????4?1
*?'
!?
inputs??????????
p
? "????????????
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_21734f????3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? ?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_21768f????3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? ?
5__inference_batch_normalization_2_layer_call_fn_21701Y????3?0
)?&
 ?
inputs????????? 
p 
? "?????????? ?
5__inference_batch_normalization_2_layer_call_fn_21714Y????3?0
)?&
 ?
inputs????????? 
p
? "?????????? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21527e?~}4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21561e?~}4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
3__inference_batch_normalization_layer_call_fn_21494X?~}4?1
*?'
!?
inputs??????????
p 
? "????????????
3__inference_batch_normalization_layer_call_fn_21507X?~}4?1
*?'
!?
inputs??????????
p
? "????????????
C__inference_city_emb_layer_call_and_return_conditional_losses_21302_</?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????-
? ~
(__inference_city_emb_layer_call_fn_21292R</?,
%?"
 ?
inputs?????????
? "??????????-?
F__inference_concatenate_layer_call_and_return_conditional_losses_21481????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
"?
inputs/9?????????
#? 
	inputs/10?????????
#? 
	inputs/11?????????
#? 
	inputs/12?????????d
#? 
	inputs/13?????????-
#? 
	inputs/14?????????2
#? 
	inputs/15?????????
#? 
	inputs/16?????????
? "&?#
?
0??????????
? ?
+__inference_concatenate_layer_call_fn_21459????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
"?
inputs/9?????????
#? 
	inputs/10?????????
#? 
	inputs/11?????????
#? 
	inputs/12?????????d
#? 
	inputs/13?????????-
#? 
	inputs/14?????????2
#? 
	inputs/15?????????
#? 
	inputs/16?????????
? "????????????
F__inference_country_emb_layer_call_and_return_conditional_losses_21336_J/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
+__inference_country_emb_layer_call_fn_21326RJ/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_1_layer_call_and_return_conditional_losses_21788^??/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? |
'__inference_dense_1_layer_call_fn_21777Q??/?,
%?"
 ?
inputs????????? 
? "???????????
B__inference_dense_2_layer_call_and_return_conditional_losses_21834^??/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
'__inference_dense_2_layer_call_fn_21824Q??/?,
%?"
 ?
inputs?????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_21661_??0?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? {
%__inference_dense_layer_call_fn_21650R??0?-
&?#
!?
inputs??????????
? "?????????? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_21803\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_21815\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? |
)__inference_dropout_1_layer_call_fn_21793O3?0
)?&
 ?
inputs?????????
p 
? "??????????|
)__inference_dropout_1_layer_call_fn_21798O3?0
)?&
 ?
inputs?????????
p
? "???????????
B__inference_dropout_layer_call_and_return_conditional_losses_21676\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_21688\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? z
'__inference_dropout_layer_call_fn_21666O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? z
'__inference_dropout_layer_call_fn_21671O3?0
)?&
 ?
inputs????????? 
p
? "?????????? ?
@__inference_model_layer_call_and_return_conditional_losses_20591?&QJC<5?~}?????????????????
???
???
0
	bathrooms#? 
	bathrooms?????????
.
bedrooms"?
bedrooms?????????
&
city?
city?????????
0
	condition#? 
	condition?????????
,
country!?
country?????????
*
floors ?
floors?????????
2

sqft_above$?!

sqft_above?????????
8
sqft_basement'?$
sqft_basement?????????
4
sqft_living%?"
sqft_living?????????
.
sqft_lot"?
sqft_lot?????????
.
statezip"?
statezip?????????
*
street ?
street?????????
&
view?
view?????????
2

waterfront$?!

waterfront?????????
,
weekday!?
weekday?????????
.
yr_built"?
yr_built?????????
6
yr_renovated&?#
yr_renovated?????????
p 

 
? "9?6
/?,
*
price!?
0/price?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_20677?&QJC<5?~}?????????????????
???
???
0
	bathrooms#? 
	bathrooms?????????
.
bedrooms"?
bedrooms?????????
&
city?
city?????????
0
	condition#? 
	condition?????????
,
country!?
country?????????
*
floors ?
floors?????????
2

sqft_above$?!

sqft_above?????????
8
sqft_basement'?$
sqft_basement?????????
4
sqft_living%?"
sqft_living?????????
.
sqft_lot"?
sqft_lot?????????
.
statezip"?
statezip?????????
*
street ?
street?????????
&
view?
view?????????
2

waterfront$?!

waterfront?????????
,
weekday!?
weekday?????????
.
yr_built"?
yr_built?????????
6
yr_renovated&?#
yr_renovated?????????
p

 
? "9?6
/?,
*
price!?
0/price?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_21049?&QJC<5?~}?????????????????
???
???
7
	bathrooms*?'
inputs/bathrooms?????????
5
bedrooms)?&
inputs/bedrooms?????????
-
city%?"
inputs/city?????????
7
	condition*?'
inputs/condition?????????
3
country(?%
inputs/country?????????
1
floors'?$
inputs/floors?????????
9

sqft_above+?(
inputs/sqft_above?????????
?
sqft_basement.?+
inputs/sqft_basement?????????
;
sqft_living,?)
inputs/sqft_living?????????
5
sqft_lot)?&
inputs/sqft_lot?????????
5
statezip)?&
inputs/statezip?????????
1
street'?$
inputs/street?????????
-
view%?"
inputs/view?????????
9

waterfront+?(
inputs/waterfront?????????
3
weekday(?%
inputs/weekday?????????
5
yr_built)?&
inputs/yr_built?????????
=
yr_renovated-?*
inputs/yr_renovated?????????
p 

 
? "9?6
/?,
*
price!?
0/price?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_21268?&QJC<5?~}?????????????????
???
???
7
	bathrooms*?'
inputs/bathrooms?????????
5
bedrooms)?&
inputs/bedrooms?????????
-
city%?"
inputs/city?????????
7
	condition*?'
inputs/condition?????????
3
country(?%
inputs/country?????????
1
floors'?$
inputs/floors?????????
9

sqft_above+?(
inputs/sqft_above?????????
?
sqft_basement.?+
inputs/sqft_basement?????????
;
sqft_living,?)
inputs/sqft_living?????????
5
sqft_lot)?&
inputs/sqft_lot?????????
5
statezip)?&
inputs/statezip?????????
1
street'?$
inputs/street?????????
-
view%?"
inputs/view?????????
9

waterfront+?(
inputs/waterfront?????????
3
weekday(?%
inputs/weekday?????????
5
yr_built)?&
inputs/yr_built?????????
=
yr_renovated-?*
inputs/yr_renovated?????????
p

 
? "9?6
/?,
*
price!?
0/price?????????
? ?
%__inference_model_layer_call_fn_20050?&QJC<5?~}?????????????????
???
???
0
	bathrooms#? 
	bathrooms?????????
.
bedrooms"?
bedrooms?????????
&
city?
city?????????
0
	condition#? 
	condition?????????
,
country!?
country?????????
*
floors ?
floors?????????
2

sqft_above$?!

sqft_above?????????
8
sqft_basement'?$
sqft_basement?????????
4
sqft_living%?"
sqft_living?????????
.
sqft_lot"?
sqft_lot?????????
.
statezip"?
statezip?????????
*
street ?
street?????????
&
view?
view?????????
2

waterfront$?!

waterfront?????????
,
weekday!?
weekday?????????
.
yr_built"?
yr_built?????????
6
yr_renovated&?#
yr_renovated?????????
p 

 
? "-?*
(
price?
price??????????
%__inference_model_layer_call_fn_20505?&QJC<5?~}?????????????????
???
???
0
	bathrooms#? 
	bathrooms?????????
.
bedrooms"?
bedrooms?????????
&
city?
city?????????
0
	condition#? 
	condition?????????
,
country!?
country?????????
*
floors ?
floors?????????
2

sqft_above$?!

sqft_above?????????
8
sqft_basement'?$
sqft_basement?????????
4
sqft_living%?"
sqft_living?????????
.
sqft_lot"?
sqft_lot?????????
.
statezip"?
statezip?????????
*
street ?
street?????????
&
view?
view?????????
2

waterfront$?!

waterfront?????????
,
weekday!?
weekday?????????
.
yr_built"?
yr_built?????????
6
yr_renovated&?#
yr_renovated?????????
p

 
? "-?*
(
price?
price??????????
%__inference_model_layer_call_fn_20819?&QJC<5?~}?????????????????
???
???
7
	bathrooms*?'
inputs/bathrooms?????????
5
bedrooms)?&
inputs/bedrooms?????????
-
city%?"
inputs/city?????????
7
	condition*?'
inputs/condition?????????
3
country(?%
inputs/country?????????
1
floors'?$
inputs/floors?????????
9

sqft_above+?(
inputs/sqft_above?????????
?
sqft_basement.?+
inputs/sqft_basement?????????
;
sqft_living,?)
inputs/sqft_living?????????
5
sqft_lot)?&
inputs/sqft_lot?????????
5
statezip)?&
inputs/statezip?????????
1
street'?$
inputs/street?????????
-
view%?"
inputs/view?????????
9

waterfront+?(
inputs/waterfront?????????
3
weekday(?%
inputs/weekday?????????
5
yr_built)?&
inputs/yr_built?????????
=
yr_renovated-?*
inputs/yr_renovated?????????
p 

 
? "-?*
(
price?
price??????????
%__inference_model_layer_call_fn_20886?&QJC<5?~}?????????????????
???
???
7
	bathrooms*?'
inputs/bathrooms?????????
5
bedrooms)?&
inputs/bedrooms?????????
-
city%?"
inputs/city?????????
7
	condition*?'
inputs/condition?????????
3
country(?%
inputs/country?????????
1
floors'?$
inputs/floors?????????
9

sqft_above+?(
inputs/sqft_above?????????
?
sqft_basement.?+
inputs/sqft_basement?????????
;
sqft_living,?)
inputs/sqft_living?????????
5
sqft_lot)?&
inputs/sqft_lot?????????
5
statezip)?&
inputs/statezip?????????
1
street'?$
inputs/street?????????
-
view%?"
inputs/view?????????
9

waterfront+?(
inputs/waterfront?????????
3
weekday(?%
inputs/weekday?????????
5
yr_built)?&
inputs/yr_built?????????
=
yr_renovated-?*
inputs/yr_renovated?????????
p

 
? "-?*
(
price?
price??????????
D__inference_reshape_1_layer_call_and_return_conditional_losses_21387\3?0
)?&
$?!
inputs?????????-
? "%?"
?
0?????????-
? |
)__inference_reshape_1_layer_call_fn_21375O3?0
)?&
$?!
inputs?????????-
? "??????????-?
D__inference_reshape_2_layer_call_and_return_conditional_losses_21404\3?0
)?&
$?!
inputs?????????2
? "%?"
?
0?????????2
? |
)__inference_reshape_2_layer_call_fn_21392O3?0
)?&
$?!
inputs?????????2
? "??????????2?
D__inference_reshape_3_layer_call_and_return_conditional_losses_21421\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? |
)__inference_reshape_3_layer_call_fn_21409O3?0
)?&
$?!
inputs?????????
? "???????????
D__inference_reshape_4_layer_call_and_return_conditional_losses_21438\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? |
)__inference_reshape_4_layer_call_fn_21426O3?0
)?&
$?!
inputs?????????
? "???????????
B__inference_reshape_layer_call_and_return_conditional_losses_21370\3?0
)?&
$?!
inputs?????????d
? "%?"
?
0?????????d
? z
'__inference_reshape_layer_call_fn_21358O3?0
)?&
$?!
inputs?????????d
? "??????????d?
#__inference_signature_wrapper_20752?&QJC<5?~}?????????????????
? 
???
0
	bathrooms#? 
	bathrooms?????????
.
bedrooms"?
bedrooms?????????
&
city?
city?????????
0
	condition#? 
	condition?????????
,
country!?
country?????????
*
floors ?
floors?????????
2

sqft_above$?!

sqft_above?????????
8
sqft_basement'?$
sqft_basement?????????
4
sqft_living%?"
sqft_living?????????
.
sqft_lot"?
sqft_lot?????????
.
statezip"?
statezip?????????
*
street ?
street?????????
&
view?
view?????????
2

waterfront$?!

waterfront?????????
,
weekday!?
weekday?????????
.
yr_built"?
yr_built?????????
6
yr_renovated&?#
yr_renovated?????????"1?.
,
dense_2!?
dense_2??????????
G__inference_statezip_emb_layer_call_and_return_conditional_losses_21319_C/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????2
? ?
,__inference_statezip_emb_layer_call_fn_21309RC/?,
%?"
 ?
inputs?????????
? "??????????2?
E__inference_street_emb_layer_call_and_return_conditional_losses_21285_5/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????d
? ?
*__inference_street_emb_layer_call_fn_21275R5/?,
%?"
 ?
inputs?????????
? "??????????d?
F__inference_weekday_emb_layer_call_and_return_conditional_losses_21353_Q/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
+__inference_weekday_emb_layer_call_fn_21343RQ/?,
%?"
 ?
inputs?????????
? "??????????