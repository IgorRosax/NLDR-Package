#include "HSLMDS_HELPERS.h"
#include <algorithm>
#include <RcppArmadillo.h>
#include <cmath>

using namespace std;
using namespace arma;
using namespace Rcpp;

arma::mat getRankedDistanceMatrix(arma::mat &data){
  arma::mat RankedDistanceMatrix(data.n_rows, data.n_cols); 
  
  for ( unsigned int i = 0; i < data.n_rows ; i++){
    RankedDistanceMatrix.row(i) = arma::sort(data.row(i));
  }
  return (RankedDistanceMatrix);
}

//' Get Neighborhood Matrix
//'
//' This function computes a neighborhood matrix based on the input data matrix. 
//' The neighborhood matrix indicates which items are neighbors to each other.
//'
//' @param data A distance matrix (dist()) or a square matrix.
//' @param k An integer specifying the number of nearest neighbors to consider.
//' @param symmetric A boolean indicating whether the neighborhood matrix should be symmetric.
//'
//' @return A binary matrix where a value of 1 indicates that the corresponding items are neighbors.
//'
//' @details
//' The function first computes a ranked distance matrix from the input data. 
//' If `symmetric` is TRUE, the neighborhood relationship is made symmetric by considering the 
//' k-nearest neighbors for both items in the pair. If `symmetric` is FALSE, the neighborhood 
//' relationship is determined only based on the k-nearest neighbors of each item individually.
//'
//' @examples
//' \dontrun{
//' data <- matrix(runif(100), nrow=10)
//' k <- 3
//' symmetric <- TRUE
//' neighborhood_matrix <- getNeighborhoodMatrix(data, k, symmetric)
//' }
//'
//' @export
// [[Rcpp::export]]
arma::imat getNeighborhoodMatrix(arma::mat &data, int k, bool symmetric){
   arma::mat RankedDistanceMatrix(data.n_rows, data.n_cols);
   arma::imat NeighborhoodMatrix (data.n_rows, data.n_cols, fill::zeros);
   
   RankedDistanceMatrix = getRankedDistanceMatrix(data);
   
   if (symmetric){
     for ( unsigned int i = 0; i < data.n_rows ; i++){
       for ( unsigned int j = 0; j <= i ; j++){
         if ( data(i,j) <= RankedDistanceMatrix(i,k) || data(i,j) <= RankedDistanceMatrix(j,k) ){
           NeighborhoodMatrix(i,j) = 1;
           NeighborhoodMatrix(j,i) = 1;
         }
       }
     }
   }
   else{
     for ( unsigned int i = 0; i < data.n_rows ; i++){
       for ( unsigned int j = 0; j < data.row(i).n_cols ; j++){
         if ( data(i,j) <= RankedDistanceMatrix(i,k) ){
           NeighborhoodMatrix(i,j) = 1;
         }
       }
     }
   }
   
   
   return (NeighborhoodMatrix);
 }


//' Get Neighborhood Vector
//'
//' This function computes a neighborhood vector based on the input data vetor. 
//' The neighborhood vector indicates which items are neighbors to each other.
//'
//' @param data A distance vector.
//' @param k An integer specifying the number of nearest neighbors to consider.
//'
//' @return A binary vector where a value of 1 indicates that the corresponding items are neighbors.
//'
//' @details
//' The function first computes a ranked distance vector from the input data. 
//' The neighborhood relationship is determined only based on the k-nearest neighbors of the 
//' vector individually.
//'
//' @examples
//' \dontrun{
//' data <- vector(runif(100), ncol=10)
//' k <- 3
//' symmetric <- TRUE
//' neighborhood_vector <- getNeighborhoodVector(data, k)
//' }
//'
//' @export
// [[Rcpp::export]]
arma::ivec getNeighborhoodVector(arma::vec &data, int k){
  arma::vec RankedDistanceVector(data.n_elem);
  arma::ivec NeighborhoodVector (data.n_elem, fill::zeros);
  
  RankedDistanceVector = arma::sort(data);
  
  for ( unsigned int i = 0; i < data.n_elem ; i++){
    if ( data(i) <= RankedDistanceVector(k) ){
      NeighborhoodVector(i) = 1;
    }
  }
  
  return (NeighborhoodVector);
}
 
 double getEuclideanDistance(arma::rowvec data1, arma::rowvec data2){
   
   double distance = arma::norm(data1 - data2, 2);
   
   return (distance);
 }
 
//' Get Euclidean Distance Matrix
//'
//' This function computes the Euclidean distance matrix for the given data matrix.
//'
//' @param data A distance matrix (dist()) or a square matrix.
//'
//' @return A symmetric matrix where each element (i, j) represents the Euclidean distance between the i-th and j-th items.
//'
//' @details
//' The function calculates the Euclidean distance between each pair of rows in the input data matrix. 
//' The resulting distance matrix is symmetric, with the distance between item i and item j being the same as the distance between item j and item i.
//'
//' @examples
//' \dontrun{
//' data <- matrix(runif(100), nrow=10)
//' distance_matrix <- getEuclideanDistanceMatrix(data)
//' }
//'
//' @export
// [[Rcpp::export]]
arma::mat getEuclideanDistanceMatrix(arma::mat &data){
   arma::mat EuclideanDistanceMatrix(data.n_rows, data.n_rows); 
   
   for ( unsigned int i = 0; i < data.n_rows ; i++){
     for ( unsigned int j = 0; j <= i ; j++){
       EuclideanDistanceMatrix(i,j) = getEuclideanDistance(data.row(i), data.row(j));
       EuclideanDistanceMatrix(j,i) = EuclideanDistanceMatrix(i,j);
     }
   }
   return (EuclideanDistanceMatrix);
 }

arma::mat getEuclideanDistanceMatrix(const arma::mat &data){
  arma::mat EuclideanDistanceMatrix(data.n_rows, data.n_rows); 
  
  for ( unsigned int i = 0; i < data.n_rows ; i++){
    for ( unsigned int j = 0; j <= i ; j++){
      EuclideanDistanceMatrix(i,j) = getEuclideanDistance(data.row(i), data.row(j));
      EuclideanDistanceMatrix(j,i) = EuclideanDistanceMatrix(i,j);
    }
  }
  return (EuclideanDistanceMatrix);
}

arma::vec getEuclideanDistanceVector(arma::mat &data, int i){
  arma::vec EuclideanDistanceVector(data.n_rows); 
  
  for ( unsigned int j = 0; j < data.n_rows ; j++){
    EuclideanDistanceVector(j) = getEuclideanDistance(data.row(i), data.row(j));
  }

  return (EuclideanDistanceVector);
}

LocalContinuityMetaCriterionResult getLocalContinuityMetaCriterion (arma::mat &data, arma::mat &conf, int Rn, int k){
  arma::imat assymmetricNeighborhoodPreserved;
  arma::vec Nk_i(conf.n_rows, fill::zeros);
  arma::mat confDist;
  running_stat<double> stats;
  LocalContinuityMetaCriterionResult result;
  
  confDist = getEuclideanDistanceMatrix(conf);
  
  assymmetricNeighborhoodPreserved = getNeighborhoodMatrix(confDist, k, false) + getNeighborhoodMatrix(data, k, false);
  
  assymmetricNeighborhoodPreserved.diag().fill(0);
  
  for(unsigned int i = 0; i < assymmetricNeighborhoodPreserved.n_rows; i++){
    for(unsigned int j = 0; j < assymmetricNeighborhoodPreserved.row(i).n_cols; j++){
      if ( assymmetricNeighborhoodPreserved(i,j) == 2 )
        Nk_i(i) = Nk_i(i) + 1;
    }
  }
  
  result.Nk = mean(Nk_i);
  result.Mk = result.Nk / k;
  result.Mk_adjusted = result.Mk - ( (double)k / (Nk_i.n_elem - 1));
  return result;
}

LocalContinuityMetaCriterionResult getLocalContinuityMetaCriterionByVector (arma::mat &data, arma::mat conf, int Rn, int k){
  arma::ivec assymmetricNeighborhoodPreserved;
  arma::vec Nk_i(conf.n_rows, fill::zeros);
  arma::vec confDistVec(conf.n_rows);
  arma::vec dataDistVec(conf.n_rows);
  running_stat<double> stats;
  LocalContinuityMetaCriterionResult result;
  
  if (data.n_rows != conf.n_rows){
    throw invalid_argument ("The number o rows must be equal between 'data' and 'conf'");
  }
  
  unsigned int nrRows = data.n_rows;
  
  for(unsigned int i = 0; i < nrRows; i++){
    confDistVec = getEuclideanDistanceVector(conf, i);
    dataDistVec = vectorise(data.row(i));
    assymmetricNeighborhoodPreserved = getNeighborhoodVector(confDistVec, k) + getNeighborhoodVector(dataDistVec, k);
    assymmetricNeighborhoodPreserved(i) = 0;
    for(unsigned int j = 0; j < nrRows; j++){
      if ( assymmetricNeighborhoodPreserved(j) == 2 )
        Nk_i(i) = Nk_i(i) + 1;
    }
  }
  
  result.Nk = mean(Nk_i);
  result.Mk = result.Nk / k;
  result.Mk_adjusted = result.Mk - ( (double)k / (Nk_i.n_elem - 1));
  return result;
}


//' Get Parameter T
//'
//' This function calculates the parameter T, which represents repulsion, for the given data.
//'
//' @param data A distance matrix (dist()) or a square matrix.
//' @param neighborhood A binary matrix indicating neighborhood relationships.
//' @param tau A numeric value representing a scaling factor.
//'
//' @return A numeric value representing the parameter T.
//'
//' @details
//' The function calculates the cardinality of the neighborhood and non-neighborhood relationships. 
//' It then computes the parameter T as the ratio of the neighborhood cardinality to the non-neighborhood cardinality, 
//' multiplied by the median of the data and the scaling factor tau.
//'
//' @examples
//' \dontrun{
//' data <- matrix(runif(100), nrow=10)
//' neighborhood <- matrix(sample(0:1, 100, replace=TRUE), nrow=10)
//' tau <- 0.5
//' parameterT <- getParameterT(data, neighborhood, tau)
//' }
//'
//' @export
// [[Rcpp::export]]
double getParameterT(arma::mat &data, arma::imat &neighborhood, double tau){

  double dataNumber, neighborhoodCardi;//, notNeighborhoodCardi;
  
  dataNumber = data.n_rows;
  
  neighborhoodCardi = (accu(neighborhood)- dataNumber);
  //notNeighborhoodCardi = (dataNumber*dataNumber) - dataNumber - neighborhoodCardi;
  
  arma::vec neighborhoodDists(neighborhoodCardi/2, fill::zeros);
  int counter = 0;
  
  for (unsigned int i = 0; i < data.n_rows; i++){
    for(unsigned int j = 0; j < i; j++){
      if(neighborhood(i,j) == 1){
        neighborhoodDists(counter) = data(i,j);
        counter++;
      }
    }
  }

  double neighborhoodMedian = median(neighborhoodDists);
  
  return ( (neighborhoodCardi/dataNumber) * neighborhoodMedian * tau );
}

arma::mat getHSfTheta(const arma::mat &conf, double &gamma){
  mat theta;
  
  theta = sqrt(pow(getEuclideanDistanceMatrix(conf),2) + pow(gamma,2));
  
  return (theta);
}

double getHSfTheta(double x, double y, double gamma){
  double theta;
  
  theta = sqrt( pow(x-y, 2) + pow(gamma, 2));
  
  return theta;
}
