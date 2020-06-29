# LeNet-5의 구현 및 성능 향상



#### 이 코드는 데이터분석캡스톤디자인 과목에서 수행한 프로젝트의 결과물입니다.



## 1. Overview



### (1) Problem



- Vanishing gradient

  

  LeNet-5는 출력층을 제외한 모든 층에서 활성화 함수로 하이퍼볼릭탄젠트를 이용한다. 하이퍼볼릭탄젠트의 도함수는 입력값이 0에서 멀어질수록 함숫값이 0으로 수렴한다. 이에 따라 그래디언트의 크기가 작아져서 신경망의 학습이 더뎌지는 문제가 발생한다. 현대의 신경망 모델은 이러한 문제를 해결하고자 렐루(ReLu), 소프트맥스(softmax) 등을 활성화 함수로 사용한다. 따라서 LeNet-5의 활성화 함수를 위 두 함수로 변경하고 그에 따른 성능의 차이를 측정한다.

  

- Covariate shift

  

  LeNet-5가 사용하는 그래디언트 기반 학습 방법은 학습률(learning rate)의 초깃값에 매우 민감하다. 예를 들어 학습률이 지나치게 크면 그래디언트가 폭발하여 학습이 제대로 되지 않는 문제가 발생한다. 또한 신경망의 구조적 특성상 각 층에 입력되는 훈련 집합의 분포가 달라지는 공변량 시프트가 발생한다. 이 역시 신경망의 학습을 저해하는 요인이다. 이 역시 신경망의 학습을 저해하는 요인이다. 배치 정규화(batch normalization)은 공변량 시프트 문제를 해결하고 신경망이 학습하는 데 있어서 학습률의 영향을 덜 받도록 하는 기법이다. LeNet-5에 배치 정규화를 적용한 후 그에 따른 성능의 차이를 측정한다.

  

- Overfitting

  

  신경망을 학습할 때 모델의 복잡도에 비해 데이터가 부족하면, 작은 값이었던 가중치들이 점차 증가하면서 과적합이 발생할 수 있다. 가중치 감쇠(weight decay)는 목적 함수에 규제 항을 두어 가중치의 영향력을 줄이는 식으로 과적합을 억제한다. 규제 항의 종류는 여러 가지이나 본 프로젝트에서는 가장 널리 쓰이는 L2놈을 이용한다. LeNet-5에 L2놈 규제 항을 적용한 후 그에 따른 성능의 차이를 측정한다.



### (2) Objective



- C++로 LeNet-5를 구현하며 stl(standard template library)을 제외한 다른 라이브러리는 사용하지 않는다.
- 구현한 LeNet-5의 분류 정확도(accuracy)가 99% 이상이 되도록 한다. 정확도 측정 시 데이터셋은 MNIST를 이용한다.
- 활성화 함수의 변경, 배치 정규화, 가중치 감쇠를 적용한 모델의 정확도와 기존 모델의 정확도를 비교 분석한다.



## 2. Schedule

| 주차      | 내용                                                         |
| --------- | ------------------------------------------------------------ |
| 2 ~ 3주차 | - 관심 주제 탐구 및 신청서 작성.                             |
| 4주차     | - DNN 구현.                                                  |
|           | - irist 데이터셋을 이용하여 성능을 측정.                     |
| 5주차     | - DNN에 (1) Xavier initialization, (2) Min-max normalization을 적용. |
|           | - MNIST 데이터셋을 이용하여 성능을 측정.                     |
| 6주차     | - LeNet-5 구현 시작.                                         |
|           | - (1) Convolution, (2) Pooling 연산 구현.                    |
| 7주차     | - 논문 리뷰: Yann LeCun, Leon Bottou, Yoshua Bengio, & Patric Haffiner (1998), Gradient-based learning applied to document recognition, Proceedings of the IEEE. |
| 8주차     | - CNN의 오차 역전파 과정의 이론적인 내용 학습.               |
| 9주차     | - matrix.h(행렬 연산을 위한 헤더) 구현.                      |
| 10주차    | - LeNet-5 구현 완료.                                         |
|           | - (1) layer.h, (2) convolutional_layer.h, (3) dense_layer.h, (4) activation_layer.h, (5) output_layer.h (6) network.h 구현. |
| 11주차    | - MNIST 데이터셋을 이용하여 성능 측정.                       |
|           | - 성능 측정 과정에서 Pooling 연산에서 버그가 발견되어 수정.  |
| 12주차    | - Tanh + MSE 모델(기준이 되는 모델; baseline model)의 성능을 측정. |
| 13주차    | - (1) 활성화 함수(softmax), (2) 손실 함수(Cross entropy), (3) 규제 항(L2 norm) 구현. |
|           | - Tanh + MSE + Regularization 모델 성능 측정.                |
|           | - ReLu + Softmax + Cross entropy 모델 성능 측정.             |
|           | - ReLu + Softmax + Cross entropy + Regularization 모델 성능 측정. |
| 14 주차   | - Batch normalization 구현.                                  |
|           | - Tanh + MSE + Batch normalization 모델 성능 측정.           |
|           | - ReLu + Softmax + Cross entropy + Batch normalization 모델 성능 측정. |
| 15주차    | - 소스코드 정리 및 보고서 작성                               |



## 3. Result(tested on MNIST)



### (1) Tanh + MSE 

- Epoch: 30, Batch: 20, Learning rate: 0.5
- Training error: 0.37%, Testing error: 1.03%

<center><img src="./img/mse, no_reg, no_norm.png"></center>



### (2) Tanh + MSE + Regularization

- Epoch: 30, Batch: 20, Learning rate: 0.5, Lambda: 0.0005
- Training error: 0.72%, Tresting error: 1.1%

<center><img src="./img/mse, reg_0.0005, no_norm.png"></center>



### (3) Tanh + MSE + Regularization

- Epoch: 30, Batch: 20, Learning rate: 0.5, Lambda: 0.001
- Training error: 1.09%, Testing error: 1.31%

<center><img src="./img/mse, reg_0.001, no_norm.png"></center>



### (4) Tanh + MSE + Batch normalization

- Epoch: 30, Batch: 20, Learning rate: 0.1
- Training error: 5.35%, Testing error: 5.04%

<center><img src="./img/mse, no_reg, yes_norm.png"></center>



### (5) ReLu + Softmax + Cross entropy

- Epoch: 30, Batch: 20, Learning rate: 0.1
- Training error: 0.04%, Testing error: 1%

<center><img src="./img/ce, no_reg, no_norm.png"></center>



### (6) ReLu + Softmax + Cross entropy + Regularization

- Epoch: 30, Batch: 20, Learning rate: 0.1, Lambda: 0.001
- Training error: 0.44%, Testing error: 1.16%

<center><img src="./img/ce, reg_0.001, no_norm.png"></center>



### (7) ReLu + Softmax + Cross entropy + Regularization

- Epoch: 30, Batch: 20, Learning rate: 0.1, Lambda: 0.01
- Training error: 0.94%, Testing error: 1.46%

<center><img src="./img/ce, reg_0.01, no_norm.png"></center>



### (8) ReLu + Softmax + Cross entropy + Batch normalization

- Epoch: 30, Batch: 20, Learning rate: 0.01
- Training error: 3.9%, Testing error: 4.1%

<center><img src="./img/ce, no_reg, yes_norm.png"></center>



## 4. Conclusion



- 구현한 모델의 에러율(error rate)이 약 1%로 논문에서 제시한 0.95%에 근접함.
- 활성화 함수를 relu, softmax로 손실 함수를 cross entropy로 변경하였을 때 학습이 빨라지는 것을 확인함.
- 가중치 감쇠, 배치 정규화를 적용했을 때 과적합이 억제되는 것을 확인함.
- 한편 배치 정규화를 적용했을 때 모델 성능이 더 향상될 것이라 예상했으나, 측정 결과 오히려 성능이 떨어진 것을 확인함.



## 5. Examples

Consturct LeNet-5 with batch normalization

```c++
vector<vector<int>> indices(6, vector<int>(16));
indices[0] = { 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1 };
indices[1] = { 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1 };
indices[2] = { 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1 };
indices[3] = { 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1 };
indices[4] = { 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1 };
indices[5] = { 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1 };

SimpleNN model;
model.add(new Conv2D({ 28, 28, 1 }, { 5, 5, 6 }, 2, Init::UNIFORM));
model.add(new BatchNorm({ 28, 28, 6 }));
model.add(new Activation(6, Activate::TANH));
model.add(new Pool2D({ 28, 28, 6 }, { 2, 2, 6 }, 2, Pool::AVG));
model.add(new Conv2D({ 14, 14, 6 }, { 5, 5, 16 }, 0, Init::UNIFORM, indices));
model.add(new BatchNorm({ 10, 10, 16 }));
model.add(new Activation(16, Activate::TANH));
model.add(new Pool2D({ 10, 10, 16 }, { 2, 2, 16 }, 2, Pool::AVG));
model.add(new Dense(400, 120, Init::UNIFORM));
model.add(new BatchNorm({ 120, 120, 1 }));
model.add(new Activation(1, Activate::TANH));
model.add(new Dense(120, 84, Init::UNIFORM));
model.add(new BatchNorm({ 84, 84, 1 }));
model.add(new Activation(1, Activate::TANH));
model.add(new Dense(84, 10, Init::UNIFORM));
model.add(new BatchNorm({ 10, 10, 1 }));
model.add(new Activation(1, Activate::TANH));
model.add(new Output(10, Loss::MSE));

int n_epoch = 30, batch = 20;
double l_rate = 0.1, lambda = 0.0;

model.fit(train_X, train_Y, l_rate, n_epoch, batch, lambda, test_X, test_Y);
```



## 6. Reports



- Report
- Demo video



## 7. Reference



- Yann LeCun, Leon Bottou, Yoshua Bengio, & Patric Haffiner (1998), Gradient-based learning applied to document recognition, Proceedings of the IEEE.
- Sergey Ioffe, Christian Szegedy (2015), Batch Normalization: Accelerating deep network training by reducing internal covariate shift, Proceedings of the ICML.
- Yiliang Xie, Hongyuan Jin, & Eric C.C. Tsang (2017), Improving the lenet with batch normalization and online hard example mining for digits recognition, Proceedings of the 
- 오일석 (2017), 기계 학습, 한빛아카데미
- 오일석 (2008), 패턴인식, 교보문고

