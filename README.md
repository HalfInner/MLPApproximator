# MLPApproximator
MLP Neural Network Learning

Educational project of function approximation. Main goal how increasing 
number of hidden layer perceptron influence on accuracy and quality of approximation.

3 function are mapped into three inputs giving 3 outputs. 

## Getting Started

### Installing

  * python3.7
  * numpy
  * matplotlib - optional

```
python -m pip install -r requirements.txt
python MLPApproximatorConsoleUI.py -h
```

### Example of usage

```
$> python MLPApproximatorConsoleUI.py -ds Examples/DataSetM5.txt -e 10
Approximator:   MLP Function Approximator
Approximator:   input number=3
Approximator:  output number=3
Approximator:  hidden number=3
Approximator:  Train on 82 samples
Approximator:  Epoch:    1/10
Approximator:   Epoch Time=0.0684s GlobalTime=0.0684s Loss=15.6%

Approximator:  Epoch:    2/10
Approximator:   Epoch Time=0.0408s GlobalTime=0.109s Loss=15.6%

(...)

Approximator:  Epoch:    9/10
Approximator:   Epoch Time=0.0731s GlobalTime=0.498s Loss=15.6%

Approximator:  Epoch:   10/10
Approximator:   Epoch Time=0.057s GlobalTime=0.555s Loss=15.6%

Approximator:   Training Time=0.555s

Approximator:  Testing:
Approximator:   Loss=15.8%

```

## Running the tests

Whole research is included into integration test. The result are saves into 'TestResults' folder. 
It takes around 1h per group. 3 groups exist.
```
python -m unittest MLPApproximatorTexst.test_integration.TestIntegration
```

## Author

* **Kajetan Brzuszczak** - *Submitted* - [HalfInner](https://github.com/HalfInner/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
