# lr-pytorch
Build a b**model

## tips
- overfit a single batch first 
```python
data = next(iter(train_loader))
```
- don't forget 
```python
model.eval()
# and
model.train()
```
- on backward, don't forget
```python
optimizer.zero_grad()
```
- don't use softmax with Crossentropyloss
- bias = False with BatchNorm
- 
```python
  x = torch.tensor([[1,2,3], [4,5,6]])
  ptint(x.view(3,2))
  print(x.permute(1, 0)) # this is transpose
```
- you must consider what the data is when using some data agmentation !(VerticalFlip on MNIST :{ )
- shuffle the mf data !
- don't forget to Normlize th data !
- for deterministic behavior 
```python 
seed = 7
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.cuda.manual_seed_all(seed)
torch.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
```
- [clipping grads](https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/) for RNNs, LSTMs, GRUs.
```python
# backward
optimizer.zero_grad()
loss.backward()
# here !
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
```
 
