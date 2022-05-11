<p align="center">
<a href="https://engineering.nyu.edu/"><img src="https://user-images.githubusercontent.com/68700549/118066006-eaf92080-b36b-11eb-9116-9f8e02a79534.png" align="center" height="100"></a>
</p>

<div align="center"> 
  
## New York University

 </div>

<div align="center"> 
  
# Distributed Training based on Amazon Sagemaker

#### Project  Report  for  ECE-GY  9143  Intro To High Performance Machine Learning


### Course  Instructor:  Dr.  Parijat Dube

#### Xingyu Pan, Binfang Ye

</div> 

<div align="center"> 
 
## INTRODUCTION
 
</div>

<div align="justify"> 
  
DNN plays an important role in AI's world. But training a DNN is time-comsuming. In this task, we explore some hardware based techniques and parallel techniques to reduce training time but the result does not compromise the accuracy. We conduct a comparison of experiment based on Amazon Sagemaker distributed library and normal data parallel training. Our experiments are runing on NYU HPC and Amazon SageMaker with using 5 different DNNs: VGG, ResNet18, MobileNet, ShuffleNet, DenseNet as the comparison. An interesting discovery from the experiment is that sagemaker provides support for distributed training based on cloud computing since the library takes advantage of gradient updates to communicate between nodes with a custom AllReduce algorithm. We come to the conclusion that distributed training based on Sagemaker provides support for speedup for most DNN training. 
  
</div>

<div align="center"> 
  
## Run on Sagemaker

</div>

<div align="justify"> 
We provide the steps to run our experiment. Parameter settings in `[sagemaker_submission.py]`, you need to:

1. Enter you AWS Access ID and Key. 
2. Adjust the S3 data path according to your own settings(in AWS)
3. You can use Sagemaker Notebook or local to run experiments

</div>

<div align="center">
  
## Paramter Pre-defined
  
</div>

When conducted our experiments, we use the same parameters as the control variables. The table below shows the main parameters we used in our experiment.
| Dataset | Epoch | Optimizer | Learning Rate | Number of Worker | Batch size |
| ------  | ----  | --------  | ------------- | ---------------- | ---------- |
| CIFAR-10|  20   | Adadelta  |     0.1       |       2          |     128    |

<div align="center">
  
## Code Structure
  
</div>

- `run_distributed.py` -- Run for distributed training
- `run_hpc.py` - Run for normal training
- `sagemaker_submission.py` - Run for SageMaker submission on Amazon AWS
- `utilization.py` - See GPU utilization for Current Machines
- `plot_hpml.py` - Show the graph of calculation result 


<div align="center">
  
## Results
  
</div>


<div align="justify"> 
  
We have the speedup comparison and accuracy comparison.

### Speedup Comparison:
The Fig. 1 shows the speed up information. We can see that when we used different tarinning hardware accleartors (e.g. GPUs), the total training times shows different. The Amazon SageMaker is able to speedup the DNN training is most case apparently.
<p align="center">
  
<img style = 'width:50%;height:auto;' src = 'results_img/Time.png'>

</p>
  

<div align="center">
 
Fig 1. shows the speed up information with the use of different hardware acclerators (e.g. GPUs).
  
</div>

### Accuracy Comparison:
The Fig. 2 shows that reduing training time by utilizing different hardware acclerators does not sacrifice the accuracy. From Fig. 2, we can observe that Amazon SageMaker can even yield a better accuracy compared with normal HPC. Therefore, we conclude that distributed training based on Sagemaker provides support speedup for most DNN training. 
  
<p align="center">
  
<img style = 'width:50%;height:auto;' src = 'results_img/Accuracy.png'>

</p>
  
<div align="center">
 
Fig 2. shows the accuracy information.
  
</div>
  
**The presentation [link](https://docs.google.com/presentation/d/1m24x9gfgI4jTMU9AvhQgptnmP4h4-QXJ-fn91tr32dE/edit?usp=sharing)**

</div>
