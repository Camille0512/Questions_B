### <font color="red">May I know the sample solution for the tasks?</font>



## Statement

1. Efficiency is not highly considered in these tasks.

   For normal computers, it will not take too much time to run these two programs.

2. Engineering standards are not used here.

3. <font color="red">The "one more question" is at the end of the **Readme.md**.</font>

4. Use **Typora** to read the readme.md will be better.



## How to use

There are two format of scripts that can be used, and they are the <font color="red">same</font>.

1. **Pure .py script**

   You can run the script directly after activating the virtual environment.

2. **Jupyter Notebook**

   You can view the result directly. Or you can download the notebook and run by yourself. The environment should be set on your own.

### Activate the virtual environment

```linux
source env/bin/activate
```



## Ideas towards the questions.

- **Question 1**

  The script offers a solution by using **Monte Carlo Simulation**. 

  **With several assumptions listed below**:

  1. The 1,000 **samples** are <font color="red">uniformly distributed</font> around the official guide price and can represent the population.

  2. The shoes with **counterfeit problems** obeys the <font color="red">normal distribution</font> around the given prices.

  3. The shoes <font color="orange">below \$30 should not be considered</font>. 

     Because that is far less than official price, which means it is far less than costs to production.

  4. Only consider the <font color="red">integer prices</font>. 

     The decimals will not change too much, just the accuracy.

  5. The **variance** between products has the <font color="red">maximum of 10</font>.
     Because in real life, the variance of the daily necessities will not be high. These commodities are *monopolistic competition market stuffs*. Furthermore, the official price is around \$48~\$68, with difference of 20. Therefore, 10 for maximum variance will be <font color="orange">reasonable</font>.

     According to the result shown by calling `variance_calculator()` function, it also shows that the maximum possible variance is 10.

  6. The **probabilities of variances** obeys <font color="red">exponential ditribution</font>.

     Exponential ditribution fits these kind of problems well.

     Since the parameter of exponential distribution is estimated according to the Poisson distribution of a related probelm, we can estimate the $\lambda$ parameter according to the probability of counterfeit around price \$30 and price \$50 as well as the maximum variance assumption.

     - For 30 dollars: $\lambda=\frac{60}{10\times 2}=3$
     - For 50 dollars: $\lambda=\frac{40}{10\times2}=2$
     - Therefore, the variance probability around \$30 and \$50 are $X_{30}\sim Exp(3)$ and $X_{50}\sim Exp(2)$.

  7. The **distributions range** <font color="red">not exceed</font> \$20.
     Since the probability exceed \$20 will be approach to 0, there is no need to analyze those price in this case.
     To be specific, the analysis of price lower than \$30 will not be included in this analysis, because that far less than the official price.

  **Solution:**

  1. **Generated certain number (<font color="red">large enough</font>) of variance** for the two price point.
  2. **Calculate the conditional probabilities** of counterfeit products on each price by making use of the <font color="orange">normal distribution</font> as well as the <font color="orange">generated variances</font>. Calculate the \$30 and \$50 separately, the result is conditional probability.
  3. **Calculate the probabilities** of counterfeit products on each price by using **Bayes equation** to get the estimated probability on each price.
  4. **Compare the probabilities** by using **graph** and **statistics** to get the <font color="orange">local optimization</font> and the <font color="orange">global optimization</font>.

  **Result**

  - The **optimized local price** is \$40, and the probability is 0.0045.

  - The **optimized global price** is \$70, and the probability is 0.0002.



- **Question 2**

  The **optimization of the loss function** is implemented by using **gradient descent**. Hence, the solution <font color="red">modify the gradient descent</font> to realize the constraints. More kinds of constraints can be developed based on the `constraint()` function.

  **Result - Equations**

  - Logistic Regression with non-negative coefficient constraint
    $$
    y=\frac{1}{1+e^{-(0.150739 x_0 + 0.006386 x_1 + 0.549349)}}
    $$

  - Logistic Regression with descending ordered coefficient constraint
    $$
    y=\frac{1}{1+e^{-(0.076619x_1+0.076619x_2+0.076619x_3+0.076619x_4+0.076619x_5+0.554631)}}
    $$

  Where:

  - $x_0$ represents explained variable **smoothness error**.
  - $x_1$ represents explained variable **symmetry error**.
  - $x_2$ represents explained variable **concave points error**.
  - $x_3$ represents explained variable **worst fractal dimension**.
  - $x_4$ represents explained variable **perimeter error**.

  

- **Further questions:**

  1. If there is no constraint, is LR a global optimization algorithm or a local one? why? 

     For non-constraint LR, since the loss function is a convex function, the optimization result should always be a global one. The estimated parameters will converge to only one value.

  2. When constraints 1) or 2) are imposed, global or local? why?

     Still the global one, because the loss function is still uses the same convex function. The constraints just affect the gradients.



