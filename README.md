# Rcpp-HSLocalMDS-Package

This repository contains the HSLocalMDS library for R, implemented in C++ and integrated using Rcpp.

## Description

HSLocalMDS implements the **Hyperbolic Smoothing Local Multidimensional Scaling (HSLocalMDS)** method for dimensionality reduction. The goal is to project high-dimensional data into a lower-dimensional space, preserving neighborhood relationships and applying hyperbolic smoothing for improved projection quality.

## Features

- Dimensionality reduction algorithms: HSLocalMDS, LocalMDS, HSMDS, and MDS
- Analytical computation of objective functions and gradients for all supported methods
- Local Continuity Meta Criterion (LCMC) for quantitative evaluation of embeddings
- Euclidean distance matrix calculation
- Flexible construction of neighborhood matrices and vectors
- High-performance C++ with seamless R integration via Rcpp

## Project Structure

- `R/`: R functions and interface exports
- `src/`: Main and auxiliary C++ implementations
- `man/`: Function documentation
- `DESCRIPTION`, `NAMESPACE`, `LICENSE`: Standard R package files

## System Requirements (Linux)

To build and use HSLocalMDS and its dependencies on Linux, you must have the following system libraries and tools installed:

```sh
sudo apt install gcc g++ cmake libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev
```

These are required for compiling C++ code and for linear algebra support (used by RcppArmadillo and roptim).

## Installation

To install the package and its dependencies locally:

```r
# In R
install.packages("Rcpp")
install.packages("RcppArmadillo")
install.packages("roptim")
install.packages("devtools")
devtools::install_github("IgorRosax/Rcpp-HSLocalMDS-Package")
```

## Example Usage

```r
library(HSLocalMDS)
# Example: U-shaped curve, (x^4+1), R^2 projected to R^1
set.seed(1)
x <- seq(0.9, 1.5, 0.05)
xx <- seq(-1, 1, 0.2) + runif(11, 0, 0.1)
x <- c(-x, x, xx)
Ccurve <- cbind(x, x^4 + 1)
d <- stats::dist(Ccurve)
dataset <- as.matrix(d)
Rn <- 1

conf <- cmdscale(d = dataset, k = Rn)
conf <- as.matrix(conf)

HSlocalMDSResult <- HSlocalMDS(
     data = as.matrix(d),
     conf = conf,
     Rn = 1,
     Kproj = 5,
     Kquality = 5,
     smallerUnitFree = 0.0001,
     selectBetterUnitFree = TRUE,
     n_t = 8,
     ratio = sqrt(10),
     applyHiperbolicSmoothing = TRUE,
     gamma = 1,
     n_gamma = 10000,
     rho = (1 / sqrt(10)),
     maxIt = 10000,
     optMethod = "CG"
)

title <- expression(paste("Blue lines connect the observation in ", R^2, " to the projection in ", R^1))
plot(x, x^4 + 1, ylim = c(0, 6), xlim = c(-6, 6), asp = 1,
           main = "U-shaped curve projected to one dimensional space",
           sub = title, cex.sub = 0.7, cex.main = 0.7)
segments(x, x^4 + 1, HSlocalMDSResult$conf, rep(0, length(x)), col = "blue")
abline(h = 0)
```

See the function documentation for details on parameters.

## Requirements

- R >= 3.5
- [Rcpp](https://cran.r-project.org/web/packages/Rcpp/index.html)
- [RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo/index.html)
- [roptim](https://cran.r-project.org/web/packages/roptim/index.html)
- [devtools](https://cran.r-project.org/web/packages/devtools/index.html) (for local installation)

## References

- [Rcpp: Seamless R and C++ Integration](https://cran.r-project.org/web/packages/Rcpp/index.html)
- [RcppArmadillo: R and Armadillo Integration](https://cran.r-project.org/web/packages/RcppArmadillo/index.html)
- [roptim: R Optimization Toolbox](https://cran.r-project.org/web/packages/roptim/index.html)
- Literature on Multidimensional Scaling and Hyperbolic Smoothing:
     - [Xavier VL, Maculan N, Pessanha JFM, e Provenza MM. ESCALONAMENTO MULTIDIMENSIONAL LOCAL: UMA ABORDAGEM VIA SUAVIZAÇÃO HIPERBÓLICA. Cadernos do IME-Série Estatística 2018; 44:37–7](https://doi.org/10.12957/cadest.2018.36483)
     - [Kruskal JB. Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. Psychometrika 1964; 29:1–27](https://doi.org/10.1007/BF02289565)
     - [Chen L and Buja A. Local multidimensional scaling for nonlinear dimension reduction, graph drawing, and proximity analysis. Journal of the American Statistical Association 2009; 104:209–19](https://doi.org/10.1198/jasa.2009.0111)

## License

See the `LICENSE` file for details.
