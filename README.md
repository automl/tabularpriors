# tabularpriors

A python module implementing interfaces for various public tabular priors.

You can use tabularpriors as a command-line-tool to pre-generate data from a prior, e.g. via
```
python -m tabularpriors --lib tabicl \
       --num_batches 1000 --batch_size 4 \
       --min_features 3 --max_features 3 \
       --max_seq_len 50 --max_classes 3 \
       --save_path tabicl_4k_50x3.h5
```
which can afterwards be loaded via
```python
from tabularpriors.dataloader import PriorDumpDataLoader
prior = PriorDumpDataLoader('tabicl_4k_50x3.h5', num_steps=20, batch_size=4, device='cpu')
```
You can also just let it create the data on-the-fly via:
```python
from tabularpriors.dataloader import TabICLPriorDataLoader
prior = TabICLPriorDataLoader(
    num_steps=20,
    batch_size=4,
    num_datapoints_max=50,
    min_features=3,
    max_features=3,
    max_num_classes=3,
    device='cpu'
)
```
You can check out `next(iter(prior))` if you want to see an example batch.

Check out `visualization_demo.ipynb` for some more examples.

### Supported Priors

- [TabICL](https://github.com/soda-inria/tabicl) (Classification)
- [TICL](https://github.com/microsoft/ticl) (Regression)

### Future work

We are planning to extend this repository by
- adding interfaces for more priors (e.g. TabPFNv1, TabForestPFN)
- enable easy mixing of different priors
- improving the storage format for pre-generated datasets
  - supporting varying number of datapoints
  - efficiently storing/loading tables with the same shape