## Distributed Training based on Amazon Sagemaker
This is a comparison of experiment based on Amazon Sagemaker distributed library and normal 
data parallel training.

Our experiments based on 5 DNNs: VGG, ResNet18, MobileNet, ShuffleNet, DenseNet.

### Run on Sagemaker
Parameter settings in `sagemaker_submission.py`, you need to:

1. Enter you AWS Access ID and Key. 
2. Adjust the S3 data path according to your own settings(in AWS)
3. You can use Sagemaker Notebook or local to run experiments

### Paramter Pre-defined
| Dataset | Epoch | Optimizer | Learning Rate | Number of Worker | Batch size |
| ------  | ----  | --------  | ------------- | ---------------- | ---------- |
| CIFAR-10|  20   | Adadelta  |     0.1       |       2          |     128    |

### Code Structure
- `run_distributed.py` -- Run for distributed training
- `run_hpc.py` - Run for normal training
- `sagemaker_submission.py` - Run for SageMaker submission on Amazon AWS
- `utilization.py` - See GPU utilization for Current Machines
- `plot_hpml.py` - Show the graph of calculation result 

### Results
Speedup Comparison:

<img style = 'width:50%;height:auto;' src = 'results_img/Time.png'>

Accuracy Comparison:

<img style = 'width:50%;height:auto;' src = 'results_img/Accuracy.png'>

presentation link: https://docs.google.com/presentation/d/1m24x9gfgI4jTMU9AvhQgptnmP4h4-QXJ-fn91tr32dE/edit?usp=sharing
