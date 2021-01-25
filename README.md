# **LeNet-5의 구현 및 성능 향상**([ENG ver](/README_ENG.md))

<br/>

본 글은 2020-1학기 캡스톤디자인 과목에서 수행한 프로젝트를 정리하기 위해 작성한 것입니다. 프로젝트의 주요 내용을 요약하여 작성했습니다. 

## **1. Overview**

### **(1) Model

![LeNet-5](/img/cnn/LeNet-5_modified.png)

위의 그림은 본 프로젝트에서 최종적으로 구현한 CNN 모델이다. LeNet-5의 구조를 그대로 따르되, 컨볼루션 층과 완전연결 층 뒤에 배치 정규화 층을 추가하였다. 또한 활성화 함수를 하이퍼볼릭탄젠트에서 ReLU로 변경하였으며, 출력 층에서는 softmax를 사용하였다. 마지막으로 손실 함수는 cross entropy로 변경하였다.

### **(2) Problems**

- Vanishing gradient

LeNet-5는 출력 층을 제외한 모든 층에서 활성화 함수로 하이퍼볼릭탄젠트를 이용한다. 하이퍼볼릭탄젠트의 도함수는 입력값이 0에서 멀어질수록 함숫값이 0으로 수렴한다. 이에 따라 그래디언트의 크기가 작아져서 신경망의 학습이 더뎌지는 문제가 발생한다. 현대의 신경망 모델은 이러한 문제를 해결하고자 렐루(ReLU), 소프트맥스(softmax) 등을 활성화 함수로 사용한다. 따라서 LeNet-5의 활성화 함수를 위 두 함수로 변경하고 그에 따른 성능의 차이를 측정한다.

- Covariate shift

Covariate shift는 신경망을 학습하는 과정에서 각 층에 입력되는 훈련 집합의 분포가 지속적으로 달라지는 문제를 말한다. 이 문제는 학습 속도를 느리게 하고, 신경망이 학습률(learning rate) 등의 초기값에 민감하게 반응하는 요인이 된다. 예를 들어 학습률이 지나치게 크면 그래디언트가 발산하여 학습이 제대로 되지 않는 현상이 발생한다. 배치 정규화(batch normalization)은 covariate shift를 해결하기 위해 고안된 방법이다. 따라서 LeNet-5에 배치 정규화를 적용한 후 그에 따른 성능 차이를 측정한다.

- Overfitting

신경망을 학습할 때 모델의 복잡도에 비해 데이터가 부족하면, 작은 값이었던 가중치들이 점차 증가하면서 과적합이 발생할 수 있다. 가중치 감쇠(weight decay)는 목적 함수에 규제 항을 두어 가중치의 영향력을 줄이는 식으로 과적합을 억제한다. 규제 항의 종류는 여러 가지이나 본 프로젝트에서는 가장 널리 쓰이는 L2놈(L2 norm)을 이용한다. LeNet-5에 L2놈 규제 항을 적용한 후 그에 따른 성능의 차이를 측정한다.

### **(3) Objectives**

- C++로 LeNet-5를 구현하며 stl(standard template library)을 제외한 다른 라이브러리는 사용하지 않는다.
- 구현한 LeNet-5의 분류 정확도(accuracy)가 99% 이상이 되도록 한다. 정확도 측정 시 데이터셋은 MNIST를 이용한다.
- 활성화 함수의 변경, 배치 정규화, 가중치 감쇠를 적용한 모델의 정확도와 기존 모델의 정확도를 비교 분석한다.

## **2. Results(tested on MNIST)**

### **(1) Tanh + MSE(baseline model)**

- Epoch: 30, Batch: 32, Learning rate: 0.02
- Training error: 0.19%, Testing error: 1.31%(original LeNet-5: 0.95%[1])

![result_baseline](/img/result_baseline.png)

### **(2) ReLU + Cross-Entropy**

- Epoch: 30, Batch: 32, Learning rate: 0.02
- Training error: 0.19%, Testing error: 1.31%(original LeNet-5: 0.95%[1])

![result_cross_entropy](/img/result_cross_entropy.png)

위는 활성화 함수를 tanh에서 ReLU와 softmax로, 손실 함수를 cross entropy로 변경한 후 측정한 결과이다. 그래프에서 볼 수 있듯이 baseline model과 굉장히 유사한 학습 과정을 관찰할 수 있었다. ReLU 함수를 사용하면 gradient vanishing 문제가 완화되고 결과적으로 학습 속도가 향상될 것이라는 예상에 부합하지 않는 결과였다. 이는 LeNet-5의 깊이가 얕기 때문에 상대적으로 gradient vanishing 문제에서 자유로웠던 탓이라 생각한다.

### **(3) ReLU + Cross-Entropy + Batch-Norm**alization

- Epoch: 20, Batch: 32, Learning rate: 0.02
- Training error: 0%, Testing error: 0.65%(original LeNet-5: 0.95%[1])

![result_batch_normalization](/img/result_batch_normalization.png)

위는 (2) 모델에서 배치 정규화를 추가한 모델이다. 앞선 모델들이 30번의 iteration 후에도 1% 이상의 error를 기록한 것에 비해, 배치 정규화를 추가한 모델은 단 3번의 iteration만에 1% 미만의 error를 기록했다. 또한 최종 error가 0.65%로 기존의 모델보다 더 향상된 결과를 얻을 수 있었다. 그러나 그래프에서 볼 수 있듯이 5번의 iteration 후에는 과적합이 발생했다.

### **(4) ReLU + Cross entropy + Batch-Normalization + L2-Norm**

- Epoch: 20, Batch: 32, Learning rate: 0.02, Lambda: 0.01
- Training error: 0.36%, Testing error: 0.68%(original LeNet-5: 0.95%[1])

![result_regularization](/img/result_regularization.png)

위는 (3) 모델에 L2-norm 규제항을 추가하여 학습시킨 결과이다. (3)과 마찬가지로 배치 정규화를 적용시켰기 때문에 매우 빠른 속도로 학습이 이루어졌다. 하지만 (3)과 달리 training error와 testing error 간 차이가 많이 줄어든 것을 확인 할 수 있는데, 이는 L2-norm이 과적합을 억제했기 때문이다. 

## **3. Conclusion**

- 구현한 모델의 error가 약 1.3%로 논문에서 제시한 0.95%에 근접함.
- 활성화 함수를 relu, softmax로 손실 함수를 cross entropy로 변경하였을 때, 성능이나 학습 과정 측면에서 큰 차이를 보이지 않았음.
- 배치 정규화를 적용했을 때 신경망의 학습 속도가 월등히 빨라지는 것을 확인함.
- L2-norm을 배치 정규화와 함께 사용하면 과적합을 억제하면서 보다 더 효율적으로 모델 학습을 할 수 있음.
- 최종적으로 완성한 모델의 error는 0.68%로 기존 모델(0.95%)보다 약 0.3% 낮아짐.

## 4. Implementation details

- 신경망에 입력되는 데이터는 모두 연속적인 1차원 메모리에 할당하여 처리했다. 이는 행렬 연산 시 caching이 용이하도록 하기 위해서이다.
- 모든 컨볼루션 연산을 GEMM(GEneral Matrix Multiplication)으로 처리하기 위해 im2col 함수를 추가하였다.
- GEMM은 C++에서 기본으로 제공하는 OpenMP를 이용하여 병렬처리가 가능하도록 하였다.
- 위와 같은 구현으로 연산 속도를 향상시킬 수 있었다(50sec / epoch, Intel i7-10750H 기준).

## **5. Examples**

- Construct LeNet-5 with batch normalization

```c++
int n_img_train = 60000;
int n_img_test = 10000;
int n_label = 10;
int img_size = 784;

float* train_X;
float* test_X;
int* train_Y;
int* test_Y;

allocate_memory(train_X, n_img_train * img_size);
allocate_memory(test_X, n_img_test * img_size);
allocate_memory(train_Y, n_img_train);
allocate_memory(test_Y, n_img_test);

ReadMNIST("train-images.idx3-ubyte", n_img_train, img_size, train_X);
ReadMNISTLabel("train-labels.idx1-ubyte", n_img_train, train_Y);
ReadMNIST("test-images.idx3-ubyte", n_img_test, img_size, test_X);
ReadMNISTLabel("test-labels.idx1-ubyte", n_img_test, test_Y);

SimpleNN model;
model.add(new Conv2d(6, 5, 2, { 28, 28, 1 }, "uniform"));
model.add(new BatchNorm2d);
model.add(new Activation("tanh"));
model.add(new AvgPool2d(2, 2));
model.add(new Conv2d(16, 5, 0, "uniform"));
model.add(new BatchNorm2d);
model.add(new Activation("tanh"));
model.add(new AvgPool2d(2, 2));
model.add(new Linear(120, "uniform"));
model.add(new BatchNorm1d);
model.add(new Activation("tanh"));
model.add(new Linear(84, "uniform"));
model.add(new BatchNorm1d);
model.add(new Activation("tanh"));
model.add(new Linear(10, "uniform"));
model.add(new BatchNorm1d);
model.add(new Activation("tanh"));

int n_epoch = 30, batch = 32;
float lr = 0.02f, decay = 0.01f;

SGD* optim = new SGD(lr, decay, "cross entropy");

model.fit(train_X, n_img_train, train_Y, n_label, n_epoch, batch, optim,
		  test_X, n_img_test, test_Y, n_label);

delete_memory(train_X);
delete_memory(test_X);
delete_memory(train_Y);
delete_memory(test_Y);
```

- Construct DNN(500 x 150 x 10) with batch normalization

```c++
int n_img_train = 60000;
int n_img_test = 10000;
int n_label = 10;
int img_size = 784;

float* train_X;
float* test_X;
int* train_Y;
int* test_Y;

allocate_memory(train_X, n_img_train * img_size);
allocate_memory(test_X, n_img_test * img_size);
allocate_memory(train_Y, n_img_train);
allocate_memory(test_Y, n_img_test);

ReadMNIST("train-images.idx3-ubyte", n_img_train, img_size, train_X);
ReadMNISTLabel("train-labels.idx1-ubyte", n_img_train, train_Y);
ReadMNIST("test-images.idx3-ubyte", n_img_test, img_size, test_X);
ReadMNISTLabel("test-labels.idx1-ubyte", n_img_test, test_Y);

SimpleNN model;

model.add(new Linear(500, 28 * 28, "uniform"));
model.add(new BatchNorm1d);
model.add(new Activation("relu"));
model.add(new Linear(150, "uniform"));
model.add(new BatchNorm1d);
model.add(new Activation("relu"));
model.add(new Linear(10, "uniform"));
model.add(new BatchNorm1d);
model.add(new Activation("softmax"));

int n_epoch = 30, batch = 32;
float lr = 0.01f, decay = 0;

SGD* optim = new SGD(lr, decay, "cross entropy");

model.fit(train_X, n_img_train, train_Y, n_label, n_epoch, batch, optim,
		  test_X, n_img_test, test_Y, n_label);

delete_memory(train_X);
delete_memory(test_X);
delete_memory(train_Y);
delete_memory(test_Y);
```

## **6. Appendices**

- [Back propagation in batch-normalized CNN](https://stnamjef.github.io/docs/Machine%20Learning/2020-08-01-Back%20propagation%20in%20batch-normalized%20CNN/)

## **7. Reference**

- [1] Yann LeCun, Leon Bottou, Yoshua Bengio, & Patric Haffiner (1998), Gradient-based learning applied to document recognition, Proceedings of the IEEE.
- [2] Sergey Ioffe, Christian Szegedy (2015), Batch Normalization: Accelerating deep network training by reducing internal covariate shift, Proceedings of the ICML.
- [3] Yiliang Xie, Hongyuan Jin, & Eric C.C. Tsang (2017), Improving the lenet with batch normalization and online hard example mining for digits recognition, Proceedings of the ICWAPR.
- [4] 오일석 (2017), 기계 학습, 한빛아카데미
- [5] 오일석 (2008), 패턴인식, 교보문고
- [6] 조준우 (2017), [CNN 역전파를 이해하는 가장 쉬운 방법](https://metamath1.github.io/cnn/index.html)
- [7] Jefkine Kafunah (2016), [Backpropagation In Convolutional Neural Networks](https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)
- [8] Roei Bahumi (2019), [Deep Learning - Cross Entropy Loss Derivative](http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html)
- [9] Kevin Zakka (2016), [Deriving the Gradient for the Backward Pass of Batch Normalization](https://kevinzakka.github.io/2016/09/14/batch_normalization/)





