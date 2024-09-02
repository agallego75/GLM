library(data.table)
n <- 100
X.dt <- data.table(x0 = rep(1,n), x1 = rnorm(n), x2 = rnorm(n))
y.dt <- data.table(y = rnorm(n))

X <- as.matrix(X.dt)
y <- as.matrix(y.dt)
ym <- mean(y)

B <- solve(t(X) %*% X) %*% t(X) %*% y

yp <- X %*% B


# Step 2: Calculate residuals
residuals <- y - yp

# Step 3: Calculate log-likelihood
n <- length(y)

sigma2 <- sum(residuals^2) / n

# Log-likelihood calculation
logLik_manual <- -n/2 * log(2 * pi * sigma2) - sum(residuals^2) / (2 * sigma2)

# Step 4: Calculate AIC
k <- length(B) + 1  # Number of parameters We add 1 becasue the derivative of the variance but is not estrict

AIC <- 2 * k - 2 * logLik_manual

print(AIC)

sigma <- sum((y-yp)^2)/(n-1)

SST <- sum((y-ym)^2) #Null Deviance
SSE <- sum((y-yp)^2) #Residual deviance
print(B)
print(paste0("Null Deviance= ", SST)
print(paste("Residual Deviance= ", SSE)


m <- glm(y~x1+x2, data = data.table(cbind(X.dt[, list(x1,x2)],y.dt)))
summary(m)

l <- lm(y~x1+x2, data = data.table(cbind(X.dt[, list(x1,x2)],y.dt)))


fisher_scoring_gaussian <- function(X, y, max_iter = 25, tol = 1e-8) {
  # X: Design matrix (with a column of 1s for the intercept)
  # y: Response vector
  # max_iter: Maximum number of iterations
  # tol: Convergence tolerance

  n <- length(y)                    # Number of observations
  p <- ncol(X)                      # Number of parameters (including intercept)
  beta <- rep(0, p)                 # Initialize beta (coefficients) to zero
  beta_old <- beta                  # To check convergence
  iter <- 0                         # Iteration counter
  
  # Iteratively update beta using Fisher Scoring
  for (iter in 1:max_iter) {
    # Calculate the linear predictor
    eta <- X %*% beta
    
    # For Gaussian, mu = eta
    mu <- eta
    
    # Calculate residuals
    residuals <- y - mu
    
    # Variance function (Gaussian family): variance is constant
    W <- diag(1, n, n)
    
    # Score function (gradient of the log-likelihood)
    score <- t(X) %*% residuals
    
    # Fisher Information Matrix (inverse of the expected Hessian)
    fisher_info <- t(X) %*% W %*% X
    
    # Update beta
    beta_new <- beta + solve(fisher_info) %*% score
    
    # Check for convergence
    if (sqrt(sum((beta_new - beta)^2)) < tol) {
      beta <- beta_new
      break
    }
    
    # Update beta for the next iteration
    beta <- beta_new
  }
  
  # Output results
  list(
    coefficients = as.vector(beta),
    iterations = iter,
    converged = iter < max_iter
  )
}

# Example Usage
# Simulate some data
set.seed(123)
n <- 100
X <- cbind(1, rnorm(n))  # Design matrix with intercept and one predictor
beta_true <- c(2, 3)     # True coefficients
y <- X %*% beta_true + rnorm(n, sd = 1)  # Response variable with noise

# Apply the Fisher Scoring algorithm
result <- fisher_scoring_gaussian(X, y)

# Display the results
result$coefficients
result$iterations
result$converged