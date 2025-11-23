# Week 2: Recommender Systems

## Table of Contents

1. [Making Recommendations](#making-recommendations)
2. [Using per-item features](#using-per-item-features)
3. [Collaborative filtering algorithm](#collaborative-filtering-algorithm)
4. [Binary labels: favs, likes and clicks](#binary-labels-favs-likes-and-clicks)

---

## Making Recommendations

This section introduces the topic of Recommender Systems, highlighting their significant commercial impact and setting up the basic framework and notation using the example of movie rating prediction.

### Commercial Importance

* **Widespread Use:** Recommender systems are used everywhere online (e.g., shopping sites like Amazon, streaming services like Netflix, food delivery apps).
* **High Value:** For many companies, a large fraction of sales and economic value is directly driven by the success of their recommender systems.
* **Academic vs. Commercial Attention:** The commercial impact of recommender systems is arguably vastly greater than the attention it receives in academia.

### Core Framework (Movie Rating Example)

The goal is to predict how users would rate movies they haven't yet watched (denoted by '?') to decide what to recommend.

| Item | Notation | Definition/Example |
| :--- | :--- | :--- |
| **Number of Users** | $n_u$ | In the example, $n_u = 4$ (Alice, Bob, Carol, Dave). |
| **Number of Items (Movies)** | $n_m$ | In the example, $n_m = 5$. |
| **Rating Indicator** | $r(i, j)$ | A binary value: $r(i, j) = 1$ if user $j$ has rated movie $i$; $0$ otherwise. |
| **Actual Rating** | $y^{(i, j)}$ | The rating (0 to 5 stars) given by user $j$ to movie $i$. (E.g., $y^{(3, 2)} = 4$). |

### Next Step

The subsequent lesson will begin developing an algorithm to predict the missing ratings. The first model will temporarily assume that **features (extra information)** about the movies (e.g., whether it is a romance movie or an action movie) are already available. Later in the notes, we will address how to build the system when these explicit movie features are not available.

---

## Using per-item features

This section details the first approach to building a recommender system: using **pre-existing item features** to create a personalized linear regression model for each user.

### Framework and Notation

**Initial Assumption:** We have pre-defined features ($X$) for each item (movie), such as $x_1$ (Romance level) and $x_2$ (Action level).
* $n_u$: Number of users (e.g., 4).
* $n_m$: Number of movies/items (e.g., 5).
* $n$: Number of features (e.g., 2).
* $r(i, j) = 1$: User $j$ has rated movie $i$.
* $y^{(i, j)}$: The actual rating given by user $j$ to movie $i$.

### The Model: Personalized Linear Regression

The system fits a separate linear regression model for each user $j$ to predict their rating for any movie $i$.

$$\text{Prediction for } y^{(i, j)} = \mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)}$$
* $\mathbf{w}^{(j)}$ and $b^{(j)}$ are the unique parameters (weights and bias) learned for user $j$.
* $\mathbf{x}^{(i)}$ is the feature vector for movie $i$.

### The Cost Function

The objective is to learn the parameters ($\mathbf{w}^{(j)}$ and $b^{(j)}$) for all users simultaneously by minimizing a regularized mean squared error cost function.

* **Cost Function for All Users ($J$):** The cost is the sum of the individual cost functions for every user.
    $$J(\mathbf{w}^{(1)}, b^{(1)}, \dots, \mathbf{w}^{(n_u)}, b^{(n_u)}) = \sum_{j=1}^{n_u} J(\mathbf{w}^{(j)}, b^{(j)})$$

* **Individual User Cost ($J(\mathbf{w}^{(j)}, b^{(j)})$):**
    $$J(\mathbf{w}^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i: r(i, j)=1} \left( (\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)}) - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{k=1}^{n} (w_k^{(j)})^2$$
  
    * The sum $\sum_{i: r(i, j)=1}$ means we only calculate the error for movies that user $j$ has actually rated.
    * The second term is standard regularization to prevent overfitting. (Note: The normalization constant $1/m^{(j)}$ is omitted for convenience, as it doesn't change the parameters at the minimum; <u>for detailed information see Bonus below)</u>.

### Bonus: *Dropping the Normalization Constant in Recommender Systems*

The term related to the number of movies rated by user $j$, $m^{(j)}$, is often dropped from the denominator of the Collaborative Filtering cost function because it is a constant scaling factor that does not affect the model's ultimate performance.

#### 1. Scaling Does Not Change the Minimum

* The overall goal is to find the parameters ($\mathbf{w}^{(j)}$ and $b^{(j)}$) that minimize the cost function $J$.
* $1/2m^{(j)}$ is a constant scaling factor determined by the training data.
* Multiplying or dividing the entire cost function by a positive constant only scales it vertically; it does not change the location of the minimum point (the optimal parameters).
    $$ \text{arg min}_{\mathbf{w}, b} \left[ J_{\text{original}}(\mathbf{w}, b) \right] = \text{arg min}_{\mathbf{w}, b} \left[ \mathbf{C} \cdot J_{\text{simplified}}(\mathbf{w}, b) \right] \quad \text{where } \mathbf{C} = \frac{1}{2m^{(j)}} \text{ is the constant.}$$

#### 2. Simplifies Optimization
* In Gradient Descent, dropping the constant $\frac{1}{2m^{(j)}}$ only scales the magnitude of the gradient. This is compensated for by adjusting the learning rate ($\alpha$).
* For Collaborative Filtering, the overall cost function $J_{\text{overall}}$ is a sum of individual user costs $J(\mathbf{w}^{(j)}, b^{(j)})$. Using different division factors ($m^{(j)}$) for every user's loss and regularization terms unnecessarily complicates the algebra for joint optimization.
* Dropping the constant leads to a cleaner, unified cost function primarily focused on minimization.

#### Comparison to Linear Regression (MSE)
The normalization term ($1/m$) is typically retained in standard Linear Regression (Mean Squared Error, MSE) for statistical and practical reasons.

| Context | Purpose of $J$ | Why $1/m$ is Kept/Dropped |
| :--- | :--- | :--- |
| Linear Regression | Evaluation and Comparison (MSE) | Kept, because it defines the average squared error (MSE), making the cost value interpretable and comparable across datasets of different sizes. |
| Recommender System | Optimization | Dropped, because $m^{(j)}$ is a constant that doesn't change the optimal parameters and unnecessarily complicates the joint cost function. |

### Next

* The current method relies on having pre-defined features ($\mathbf{x}^{(i)}$) for every item.
* The next section will explore a modification of this algorithm—**Collaborative Filtering**—which works even when these detailed item features are not available beforehand.

---

## Collaborative filtering algorithm

This section introduces **Collaborative Filtering**, a powerful technique for recommender systems where the item features ($\mathbf{x}$) are learned from the user ratings rather than being provided in advance.

### The Challenge: Learning Item Features ($\mathbf{x}$)
In the previous model, we assumed movie features ($\mathbf{x}$) were known (e.g., Romance, Action level). In the new new approach, when features are unknown, the ratings provided by multiple users on the same item can be leveraged to learn what those item features ($\mathbf{x}$) should be.

**Why it Works:** Having ratings from several users (each with known preference parameters $\mathbf{w}$ and $b$) allows the system to infer the features of an unfeatured movie that best explain those ratings. This relies on the "collaboration" of ratings from multiple users on the same item, which defines algorithm's name.

### Cost Function for Learning Features ($\mathbf{x}$)

If the user preference parameters ($\mathbf{w}^{(j)}, b^{(j)}$) are temporarily fixed, the features for a single movie $i$ ($\mathbf{x}^{(i)}$) are learned by minimizing the cost function:

$$\min_{\mathbf{x}^{(i)}} J(\mathbf{x}^{(i)}) = \frac{1}{2} \sum_{j: r(i, j)=1} \left( (\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)}) - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{k=1}^{n} (x_k^{(i)})^2$$

### The Full Collaborative Filtering Cost Function

The final algorithm combines the objective of learning user preferences ($\mathbf{w}, b$) and learning item features ($\mathbf{x}$) into a single unified cost function ($J$):

* **Minimization:** The algorithm simultaneously minimizes $J$ with respect to all parameters: the user parameters ($\mathbf{w}^{(j)}, b^{(j)}$ for all users $j$) and the movie features ($\mathbf{x}^{(i)}$ for all movies $i$).
* **Unified Cost ($J$):** This combines the prediction error and the regularization terms for both users and movies.

$$J(\mathbf{w}, \mathbf{b}, \mathbf{x}) = \frac{1}{2} \sum_{(i, j): r(i, j)=1} \left( (\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)}) - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2$$

### Optimization

Gradient Descent or other optimization algorithms are used to minimize the cost function $J$. In this full formulation, both the user preferences ($\mathbf{w}, \mathbf{b}$) and the item features ($\mathbf{x}$) are treated as parameters to be learned and are updated iteratively.

$$
\begin{aligned}
w_i^{(j)} &= w_i^{(j)} - \alpha \frac{\partial}{\partial w_i^{(j)}}J(w, b,x) \\
b^{(j)} &= b^{(j)} - \alpha \frac{\partial}{\partial b^{(j)}}J(w, b,x) \\
x_k^{(i)} &= x_k^{(i)} - \alpha \frac{\partial}{\partial x_k^{(i)}}J(w, b,x)
\end{aligned}
$$

### Next

The next section will address a generalization of this model to systems using binary labels (e.g., like/dislike) instead of continuous star ratings.

---

## Binary labels: favs, likes and clicks

This section explains how to adapt the collaborative filtering algorithm from predicting continuous ratings (like 1–5 stars) to predicting **binary labels** (like/dislike, purchase/not purchase), using a method analogous to moving from linear regression to logistic regression.

### Binary Label Context

Many recommender systems deal with binary labels (1 or 0) rather than star ratings.

* **Label Meanings (Engagement):**
    * **1 (Engaged):** User liked, purchased, favorited, clicked, or spent a minimum time (e.g., 30 seconds) on an item after exposure.
    * **0 (Not Engaged):** User did not like, did not purchase, or left quickly after being exposed to the item.
    * **? (Question Mark):** The user was not yet exposed to the item (no rating/engagement data).
* **Goal:** Predict the probability that a user will like or engage with a new item (the '?' items) to decide what to recommend.

### The Model: Logistic Regression Analogy

The model shifts from predicting a numerical rating to predicting a probability of engagement. The linear combination of user preferences ($\mathbf{w}^{(j)}$) and item features ($\mathbf{x}^{(i)}$) is passed through the logistic function ($g$) (also known as the sigmoid function).

$$\text{P}(y^{(i, j)}=1) = g(\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)})$$

where $g(z) = \frac{1}{1 + e^{-z}}$.

### The Cost Function: Binary Cross-Entropy

Since the output is a probability and the labels are binary, the squared error cost function (used for ratings) is replaced with the Binary Cross-Entropy Loss (or log loss), which is standard for logistic regression.

* **Loss for a Single Example:**
    $$L(f(\mathbf{x}), y) = -y \log(f(\mathbf{x})) - (1-y) \log(1-f(\mathbf{x}))$$
  
* **Overall Binary Collaborative Filtering Cost ($J$):** The total cost function sums this binary cross-entropy loss over all user-item pairs where a rating/engagement exists ($r^{(i, j)}=1$), plus the regularization terms for all $\mathbf{w}$, $\mathbf{b}$, and $\mathbf{x}$.

$$J(\mathbf{w}, \mathbf{b}, \mathbf{x}) = \sum_{(i, j): r(i, j)=1} L(f(x^{(i)}), y^{(i,j)}) + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2$$

### Generalization

This generalization significantly opens up the set of applications that can be addressed by collaborative filtering, allowing the algorithm to work with implicit feedback (like clicks or viewing time) rather than requiring explicit user ratings.