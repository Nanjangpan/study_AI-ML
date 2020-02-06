# Ch 5. Support Vector Machine

**복잡한 분류문제, 작거나 중간 크기의 데이터셋**

* Decision Boundary with Margin
* Maximizing the Margin
* Error Handling in SVM-> Soft Margin with SVM
* Kernel Trick...(Primal and Dual problem)
* SVM with Kernel



---

## Decision Boundary with Margin

주어진 샘플의 종류를 잘 구분할 수 있는 경계

어떤 하나의 값이 아님. 고차원에서 샘플을 분류해내는 경계도 됨. 

![](.\images\캡처.PNG)



확률적인 방법(나이브 베이즈 등)을 제외하고 decision boundary를 정한다면 how? 

---

![](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205012835099.png)

* Support Vector Machine

![image-20200205013152216](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205013152216.png)

### Decision boundary line

$$
wx + b  = 0\\
(x_1, x_2, b)
$$

![image-20200205020453505](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205020453505.png)


$$
positive \; case : wx + b > 0 \\
negative \; case : wx + b < 0
$$

$$
confidence\; level\\
positive\;case : (wx+b)y =>\; ++\\
negative\;case : (wx+b)y =>\; --\\
\;\\
confidence\;level >0
$$
confidence level을 최대한 높이는 w와 b를 찾아야 함.

**Margin**

![image-20200205022535036](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205022535036.png)



---

## Maximizing the Margin

![image-20200205022936907](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205022936907.png)
$$
f(x) = wx + b\\
point \;x_p \;->
f(x) = wx+b = 0
$$

$$
positive\;point\;x\\
f(x) = wx + b = a, a>0
$$


$$
distance\\
x = x_p + r\frac{w}{||w||}, f(x_p) = 0\\
$$
![image-20200205025533668](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205025533668.png)

위의 x를 f(x) 안에 집어 넣으면...
$$
f(x) = w·x + b = w(x_p + r\frac{w}{||w||}) +b = wx_p + b+r\frac{w·w}{||w||} = r||w||\\
*wx_p+b = 0\\
\,\\
f(x) = r||w|| =a
\,\\
\,\\
distance\; r = \frac{f(x)}{||w||} = \frac{a}{||w||}
$$


결국 good decision boundary를 위해서는 w가 중요

![image-20200205030722903](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205030722903.png)

### 2r의 최적화 문제

$$
max_{x,b}2r = \frac{2a}{||w||}\\
조건은\, (wx_j+b)y_j \geq a, j=instance(data)
\,\\
a는\, arbitrary\;number이므로\; 1로\;정해도\;됨\\
max_{x,b}2r = \frac{2}{||w||}이\;됨.\\
\,\\
…\\
\;\\
minimize\;w의\;문제로 \;변환\\
min_{w,b}||w||, 조건\; (wx_j+b)y_j \geq 1
$$



**||w||가 quadratic optimization문제가 되는 이유**
$$
||w|| \\
w=> w_1, w_2\\
\;\\
||w|| = \sqrt{w_1^2+w_2^2}\\
square\;problem
$$

*good quadsolution

![image-20200205034033529](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205034033529.png)



*bad quadsolution

![image-20200205034334068](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205034334068.png)

![image-20200205034147501](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205034147501.png)

왜 bad = w를 찾지 못하냐 => SVM with Hard Margin으로 error를 인정하지 않기 때문.

error를 어느 정도 인정하면 SVM with Soft Margin

* SVM Hard Margin
* SVM Soft Margin
* SVM with Kernel(Hard Margin)

---

# Error Handling in SVM

![image-20200205035057702](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205035057702.png)

### Error handling

1. **complex decision boundary(->Kernel Trick)**

2. error penalization(-> Soft Margin SVM)



### Error penalization

1. counting errors
   $$
   min_{w,b}||w|| + C*N_{error}
   $$
   ![image-20200205040501720](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205040501720.png)

0-1 Loss : quadratic problem으로 정의 어려움, decision boundary 너머의 점들에 대해 똑같은 penalty 부여



**Hinge Loss**

![image-20200205040906408](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205040906408.png)
$$
\zeta_j = slack \;variable, >1\; when\; mis-classified
\;\\
min_{w,b}||w|| + C\sum_j\zeta_j\\
조건\; (wx_j+b)y_j \geq 1-\zeta_j\;,일부\;케이스에서는\;\zeta값이\;있으므로.기본적으로\;\zeta\geq0\\
hard\;margin\;SVM : (wx_j+b)y_j \geq 1\\
$$
C: slack variable을 어느 정도의 강도로 설명할 것인지, trade-off parameter

summation은 minimum

---

# Soft Margin with SVM

![image-20200205042723057](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205042723057.png)
$$
min_{w,b}||w|| + C\sum_j\zeta_j\\
조건\; (wx_j+b)y_j \geq 1-\zeta_j
$$
점이 decision boundary를 넘어가도 되는데 넘어갈 때마다 penalize된다=soft margin SVM

* Log-loss, hinge loss, 0-1 loss

  ![image-20200205043404600](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205043404600.png)

  -log loss는 안전한 사이드에 있어도 penalty를 줌. 

  -hinge는 안전한 사이드를 완전하게 신뢰

  

**C값이 변화하면서 바뀌는 decision boundary**

![image-20200205043817592](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205043817592.png)

-무조건 C가 크면 좋다? No. C=10, C=10000 큰 차이 없음. ==> 결국 실제 적용은 EDA 부분에서 조절해줘야 함.





---

# Kernel Trick

-non linear가 일관된 trend가 있다고 하면.. 즉, 데이터가 complex하다면..



### Error handling

1. complex decision boundary(->Kernel Trick)

2. **error penalization(-> Soft Margin SVM)**



![image-20200205044303308](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205044303308.png)

![image-20200205045018659](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205045018659.png)

-linear regression(higher dimension) : interaction terms
$$
X_1, X_2, X_1^2, X_2^2, X_1X_2, X_1^3, X_2^3...
$$
-higher dimension with SVM? 

Primal problem-> Dual problem

![image-20200205045400242](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205045400242.png)

SVM

: Classification-> Constrained quadratic programming, Constrained optimization



**라그랑주 승수법**

(강의랑 조금 다름)

-제약이 있는 최적화 문제를 풀 때 사용, 람다는 나중에 없어지는 변수
$$
제약조건: g(x,y)\\
최적화 함수: f(x,y)\\
\;\\
1. g(x,y) = 0일 때 \lambda g(x,y) = 0\\
2. L = f(x,y) - \lambda g(x,y)일 때 L(x,y,\lambda)의\;최댓값과\;최솟값은 \;g(x,y)=0일때\\
\; f(x,y)의\;최댓값과\;최솟값을\;의미함\\
3. \frac{\partial L}{\partial x}= 0, \frac{\partial L}{\partial y}= 0, \frac{\partial L}{\partial \lambda}= 0일때\; L은\;최댓값이나\;최솟값(극값)을\;가짐.\\
4. 정리=> \nabla L = (\frac{\partial L}{\partial x}, \frac{\partial L}{\partial y}, \frac{\partial L}{\partial \lambda}) = \lim 0
$$
![image-20200205051548128](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205051548128.png)

예시)

![image-20200205051644452](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205051644452.png)

유도방법: https://untitledtblog.tistory.com/96

라그랑주 설명(외국사이트): http://www.slimy.com/~steuard/teaching/tutorials/Lagrange.html



강의
$$
constrained \; optimization: min_xf(x)\\
condition: g(x) \leq 0, h(x) = 0 \\
Lagrange\;Prime\;Function: L(x,\alpha, \beta) = f(x) + \alpha g(x) + \beta h(x)\\
Lagrange\;Multiplier: \alpha \geq 0, \beta\\
Lagrange\;Dual\;Function: d(\alpha, \beta) = inf_{x \in X}L(x,\alpha,\beta)=min_xL(x,\alpha, \beta)
$$
![image-20200205052930437](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205052930437.png)



->primal problem을 dual problem으로 풀겠다가 핵심

![image-20200205053857866](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205053857866.png)

![image-20200205054036126](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205054036126.png)

Dual Problem의 속성

-Weak duality theorem: 최대화 문제면 primal problem의 최적해에 대한 상한이 dual problem의 값이다.

-Strong duality: dual problem에서 최소 상한값이 primal problem에서의 최적값과 같다. 이때 KKT조건이 만족되어야 함.



-KKT

![image-20200205054807294](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205054807294.png)



![image-20200205055146243](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205055146243.png)

![image-20200205055213252](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205055213252.png)

![image-20200205055229040](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205055229040.png)



KKT 만족하는지 w,b에 대해 미분해서 확인

![image-20200205055551765](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205055551765.png)

![image-20200205055914441](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205055914441.png)
$$
\alpha의 \;2차식으로 \;돌아옴=quadratic\;programming
$$
![image-20200205060055089](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205060055089.png)



**Mapping Function**

linearly unseparable-> separable

![image-20200205060402756](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205060402756.png)

편리, but too much interaction(dimensions)----> Kernel solution



---



### Kernel Function

![image-20200205060650814](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205060650814.png)

K=내적, 각자 다른 dimension



커널 종류

-선형

-다항식

-가우시안

-시그모이드



x와 z를 먼저 내적해서 3차로 하는 것(Kernel trick, 연산량 작음)= x와 z를 다른 차원으로 보내서 그걸 내적하는 것(Kernel, 연산량 폭증)



![image-20200205061838017](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205061838017.png)

![image-20200205061914156](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205061914156.png)



**SVM Kernel trick**

![image-20200205062517280](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205062517280.png)



<정리>

Hard Margin(오류)--> Soft Margin(정확하지 X)-->Kernel(연산량 폭증)--> Kernel Trick







------

# Ch 6. Evaluation(Train/Test, Regularization)



* 온라인 학습
* Error 값
* CV
* Regularization



-----



### 온라인 학습

: 데이터를 순차적으로 한 개씩 또는 **미니배치**mini-batch 라 부르는 작은 묶음 단위로 주입하여 시스템을 훈련시킵니다. 매 학습 단계가 빠르고 비용이 적게 들어 시스템은 데이터가 도착하는 대로 즉시 학습할 수 있습니다(그림 1 -13).연속적으로 데이터를 받고(예를 들면 주식가격) 빠른 변화에 스스로 적응해야 하는 시스템에 적합합니다. 컴퓨팅 자원이 제한된 경우에도 좋은 선택입니다. 온라인 학습 시스템이 새로운 데이터 샘플을 학습하면 학습이 끝난 데이터는 더 이상 필요하지 않으므로 버리면 됩니다(이전 상태로 되돌릴 수 있도록 데이터를 재사용하기 위해 보관할 필요가 없다면). 이렇게 되면 많은 공간을 절약할 수 있습니다.(핸즈온머신러닝, 2018)

![스크린샷 2018-05-24 오후 6.20.09](https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-24-e1848be185a9e18492e185ae-6-20-09.png?w=625)

![스크린샷 2018-05-24 오후 6.23.50](https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-24-e1848be185a9e18492e185ae-6-23-50.png?w=625)



---

### Error값

![image-20200205064739576](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205064739576.png)

• f: the target function to learn

• g: the learning function of ML 

• $g^(D)$: the learned function by using a dataset, D, or an instance of hypothesis 

• D: an available dataset drawn from the real world

 • $\overline g$: the average hypothesis of a given infinite number of Ds



![image-20200205071809892](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205071809892.png)

variance-bias : trade off relation

bias: 모델을 더 복잡하게. fitting-> overfitting



안전하고 정확도 낮은 모델 vs 위험하고 일부 정확도 높은 모델=> trade off

![image-20200205073816185](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205073816185.png)

complex model(right) : high variance

simple model(left) : high bias



오컴의 면도날: 동일한 모델이면 low complexity를 가진 것으로 선택



---

### Cross Validation

: mimic infinite number of sampling

![image-20200205074331534](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205074331534.png)

![image-20200205074358601](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205074358601.png)

LOOCV: 하나 남겨두고 모두 테스팅 * N개(Leave One Out Cross Validation)



* Precision/Recall/F1 measure

![image-20200205074608671](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205074608671.png)

Precision = $\frac{TP} {(TP+FP)}$ = 예측한 것 중에 얼마나 맞췄는지

Recall = $\frac {TP} { (TP+FN)}$ = 답 중에 얼마나 맞췄는지

F1 -Measure= $2 \frac {(Precision * Recall)} {(Precision + Recall)}$



* Fb -Measure = $(1+b^2 ) \frac {(Precision * Recall)} {(b^2 Precision + Recall)}$





---

### Regularization

: perfect fit을 포기함. test set의 potential을 높이기 위함.



E(w) = error term

![image-20200205080105587](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205080105587.png)

![image-20200205080241959](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205080241959.png)

Ridge가 더 보편적(파라미터가 0이 되진 않음). 다 반영할 수 있음. 

Lasso는 많은 파라미터가 날라감(0이 됨). 몇몇 features 만 반응. 반응 빠름.



Regularization effect

![image-20200205080928188](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205080928188.png)



![image-20200205081102574](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205081102574.png)



SVM Regularization => C parameter

![image-20200205081320996](C:\Users\thest\AppData\Roaming\Typora\typora-user-images\image-20200205081320996.png)
