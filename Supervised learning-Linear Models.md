Linear Models
==
    {y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p

* Y	: predicted value
* W1~Wp : coef_
* W0 	: intercept_

Ordinary Least Squares(普通最小平方)
--
* LinearRegression fits a linear model with coefficients, to minimize the cost function
* Multicollinearity : 意味著變數之間有高度重複性造成的異常(變數裡有身高、體重、BMI會影響其他變數的表現)
> Code:

        reg = linear_model.LinearRegression() 
        讓 reg 去 fit 某些data，並製造一個Linear model，並存取coefficients
        reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
        reg.coef_ => 列出目前的coef array([0.5, 0.5])
        
* Non-Negative Least Squares: 如果常數不可能為負，可以在過程中添加限制來達到目的
> Code:
        
        reg_nnls = LinearRegression(positive=True)

* Ordinary Least Squares Complexity: 如果Sample > Features: O( Nsample * Nfeatures**2)

Ridge regression and classification
-- 
* Ridge regression: 由於OLS造成的Overfitting，藉由增添 λΣθ² 來避免(L2范数)
> Code:

        reg = linear_model.Ridge(alpha=.5)
        reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) # Ridge(alpha=0.5)
        reg.coef_               # array([0.34545455, 0.34545455])
        >>> reg.intercept_      # 0.13636...

* Ridge Classification: 處理Classification的資料，相較於LogisticRegression 快
* Ridge Complexity: 與OLS相同O( Nsample * Nfeatures**2)
* Setting the regularization parameter: leave-one-out Cross-Validation
> Leave-one-out Cross-Validation: 將每個資料點輪流當成test，剩餘當成train，花費n-1倍的時間

> K-fold cross validation: 資料隨機分成k組，每次拿一組當成test，剩餘組當成train

> RidgeCV : 藉由一次用多個alpha來尋找最佳解

Lasso 
--
* Lasso regression: 與Ridge相同，藉由增添 λΣθ 來避免(L1范数)
> Code:
    from sklearn import linear_model
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit([[0, 0], [1, 1]], [0, 1])   #  Lasso(alpha=0.1)
    reg.predict([[1, 1]])               #  Array([0.8])
  
* Setting the regularization parameter: Using cross-validation
* LassoLarsIC: 尋找最佳 alpha，並減少計算過程(從K+1 => 1)
* Comparison with the regularization parameter of SVM
