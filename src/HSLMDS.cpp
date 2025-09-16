// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(roptim)]]
#include "HSLMDS_HELPERS.h"
#include <algorithm>
#include <RcppArmadillo.h>
#include <roptim.h>

using namespace std;
using namespace arma;
using namespace roptim;



//' Get Local MDS Stress
//'
//' This function calculates the local Multidimensional Scaling (MDS) stress for the given data.
//'
//' @param data A distance matrix (dist()) or a square matrix.
//' @param confVec A numeric vector representing the configuration vector.
//' @param neighborhood A binary matrix indicating neighborhood relationships.
//' @param notNeighborhood A binary matrix indicating non-neighborhood relationships.
//' @param Rn An integer specifying the number of dimensions for the configuration.
//' @param tt A numeric value representing the weight for the non-neighborhood stress.
//' @param Gamma A numeric constant value used for smoothing the configuration.
//'
//' @return A numeric value representing the local MDS stress.
//'
//' @details
//' The function first reshapes the configuration vector into a matrix and computes the Euclidean distance matrix for the configuration. 
//' It then calculates the difference between the original data and the configuration distance matrix, squares these differences, and 
//' multiplies them by the neighborhood matrix. The non-neighborhood stress is calculated similarly but using the non-neighborhood matrix.
//' The final stress value is the difference between the neighborhood stress and the weighted non-neighborhood stress.
//'
//' @examples
//' \dontrun{
//' data <- matrix(runif(100), nrow=10)
//' confVec <- runif(30)
//' neighborhood <- matrix(sample(0:1, 100, replace=TRUE), nrow=10)
//' notNeighborhood <- matrix(sample(0:1, 100, replace=TRUE), nrow=10)
//' Rn <- 3
//' tt <- 0.5
//' stress <- getLocalMdsStress(data, confVec, neighborhood, notNeighborhood, Rn, tt)
//' }
//'
//' @export
// [[Rcpp::export]]
double getHSLocalMdsStress (arma::mat &data,const arma::vec &confVec, arma::imat &neighborhood,arma::imat &notNeighborhood, unsigned int &Rn, double &tt, double &Gamma){
  
  arma::mat confDist;
  
  double neighborhoodStress = 0;
  double notNeighborhoodStress = 0;
  
  arma::mat conf = reshape(confVec, Rn, data.n_rows);
  conf = conf.t();
  
  confDist = getHSfTheta(conf, Gamma);
  
  for (unsigned int i = 0; i < neighborhood.n_rows - 1; i++)
  {
    for (unsigned int j = i + 1; j < neighborhood.n_rows; j++)
    {
      neighborhoodStress += pow( data(i, j) - confDist(i, j), 2 ) * neighborhood(i, j);
      notNeighborhoodStress += confDist(i, j) * notNeighborhood(i, j);
    }
  }
  return( neighborhoodStress - (tt * notNeighborhoodStress));
}

//' Get HS Local MDS Stress Gradient
//'
//' This function calculates the gradient of the local Multidimensional Scaling (MDS) stress for the given data.
//'
//' @param data A distance matrix (dist()) or a square matrix.
//' @param confVec A numeric vector representing the configuration vector.
//' @param neighborhood A binary matrix indicating neighborhood relationships.
//' @param notNeighborhood A binary matrix indicating non-neighborhood relationships.
//' @param Rn An integer specifying the number of dimensions for the configuration.
//' @param tt A numeric value representing the weight for the non-neighborhood stress.
//' @param Gamma A numeric constant value used for smoothing the configuration.
//'
//' @return A numeric vector representing the gradient of the local MDS stress.
//'
//' @details
//' The function reshapes the configuration vector into a matrix and computes the Euclidean distance matrix for the configuration. 
//' It then calculates the difference between the original data and the configuration distance matrix, normalizes these differences, 
//' and computes the gradient based on the neighborhood and non-neighborhood relationships. The final gradient is returned as a vector.
//'
//' @examples
//' \dontrun{
//' data <- matrix(runif(100), nrow=10)
//' confVec <- runif(30)
//' neighborhood <- matrix(sample(0:1, 100, replace=TRUE), nrow=10)
//' notNeighborhood <- matrix(sample(0:1, 100, replace=TRUE), nrow=10)
//' Rn <- 3
//' tt <- 0.5
//' gradient <- getHSLocalMdsStressGradient(data, confVec, neighborhood, notNeighborhood, Rn, tt)
//' }
//'
//' @export
// [[Rcpp::export]]
arma::vec getHSLocalMdsStressGradient(arma::mat &data,const arma::vec &confVec, arma::imat &neighborhood,arma::imat &notNeighborhood, unsigned int &Rn, double &tt, double &Gamma){
  
  arma::mat confDist;
  arma::mat distDif;
  arma::mat parcela;
  arma::mat grad(neighborhood.n_rows, Rn, arma::fill::zeros);
  arma::mat conf = reshape(confVec, Rn, data.n_rows);
  
  conf = conf.t();
  
  confDist = getHSfTheta(conf, Gamma);
  
  distDif = (data - confDist);

  for(unsigned int i = 0; i < neighborhood.n_rows - 1; i++){
    for(unsigned int j = i + 1; j < neighborhood.n_rows; j++){
      grad.row(i) = grad.row(i) + ((+2) * neighborhood(i,j) * distDif(i,j) * (conf.row(i) - conf.row(j)) / confDist(i,j)) +
        (tt * (notNeighborhood(i,j)/confDist(i,j)) * (conf.row(i) - conf.row(j)));
      grad.row(j) = grad.row(j) + ((+2) * neighborhood(j,i) * distDif(j,i) * (conf.row(j) - conf.row(i)) / confDist(j,i)) +
        (tt * (notNeighborhood(j,i)/confDist(j,i)) * (conf.row(j) - conf.row(i)));
    }
  }
  
  return vectorise(-1*(grad.t())); 
}

class optimHSLocalMdsStress : public Functor{
public:
  optimHSLocalMdsStress(
    arma::mat &data,
    arma::mat &conf,
    arma::imat &neighborhood,
    arma::imat &notNeighborhood
  ) : data(data), conf(conf), neighborhood(neighborhood), notNeighborhood(notNeighborhood) {}
  
  double operator()(const arma::vec &x ) override {
    return getHSLocalMdsStress(data, x, neighborhood,notNeighborhood, Rn, tt, Gamma);
  }
  
  void Gradient(const arma::vec &x, arma::vec &grad) override {
    grad = getHSLocalMdsStressGradient(data, x, neighborhood,notNeighborhood, Rn, tt, Gamma);
    
  }
  arma::mat getData (){return data;}
  arma::mat getConf (){return conf;}
  arma::imat getNeighborhood (){return neighborhood;}
  arma::imat getNotNeighborhood (){return notNeighborhood;}
  int getRn () {return Rn;}
  double getTt () {return tt;}
  
  void setData (arma::mat &data){this->data = data;}
  void setConf (arma::mat &conf){this->conf = conf;}
  void setNeighborhood (arma::imat &neighborhood){this->neighborhood = neighborhood;}
  void setNotNeighborhood (arma::imat &notNeighborhood){this->notNeighborhood = notNeighborhood;}
  void setRn (unsigned int Rn) {this->Rn = Rn;}
  void setTt (double tt) {this->tt = tt;}
  void setGamma (double Gamma) {this->Gamma = Gamma;}
private:
  arma::mat &data;
  arma::mat &conf;
  arma::imat &neighborhood;
  arma::imat &notNeighborhood;
  unsigned int Rn = 0;
  double tt = 0;
  double Gamma = 0;
};


optimResult cppOptimHSLocalMds (arma::mat &data, arma::mat &conf, arma::imat &neighborhood,arma::imat &notNeighborhood, 
                             unsigned int &Rn, double &tt, double &Gamma, 
                             int &maxIt, const std::string &optMethod, unsigned int trace, unsigned int optReport){
  optimHSLocalMdsStress optimStress (data, conf, neighborhood, notNeighborhood);
  Roptim<optimHSLocalMdsStress> opt(optMethod);
  optimResult result;
  
  optimStress.setRn(Rn);
  optimStress.setTt(tt);
  optimStress.setGamma(Gamma);
  
  opt.control.maxit = maxIt;
  opt.control.trace = trace;
  opt.control.REPORT = optReport;
  
  arma::vec minimizeParameter = vectorise(conf);
  opt.minimize(optimStress, minimizeParameter);
  
  result.parameter = reshape(minimizeParameter, conf.n_rows, conf.n_cols);
  result.value = opt.value();
  result.fnCount = opt.fncount();
  result.grCount = opt.grcount();
  result.convergence = opt.convergence();
  result.message = opt.message();
  
  return result;
}

HsLocalMdsResult cppHSlocalMDS (arma::mat &data,
                             arma::mat conf,
                             unsigned int Rn = 2,
                             unsigned int Kproj = 5,
                             unsigned int Kquality = 0,
                             bool verbose = false,
                             bool selectBetterUnitFree = false,
                             double smallerUnitFree = 0.0001,
                             unsigned int n_t = 10,
                             double ratio = 3.162278,
                             bool applyHiperbolicSmoothing = true,
                             double gamma = 1,
                             unsigned int n_gamma = 30,
                             double rho = 0.5,
                             int maxIt = 30,
                             const std::string optMethod = "CG",
                             unsigned int optTrace = 0, 
                             unsigned int optReport = 10){
  
  double smallerGammaAllowed;
  smallerGammaAllowed = pow(10, -16);
  
  
  if( !data.is_square() )
    throw invalid_argument("distances must be result of 'dist' or a square matrix");
  
  if ( conf.n_rows!=data.n_rows || conf.n_cols!=Rn ) 
    throw invalid_argument("The number of rows of the 'x' must be the same of the numbers of observations of the data set. The number of columns must be the same of the 'Rn'");
  
  if(Kproj >= data.n_rows)
    throw invalid_argument ("The 'Kproj' must be lower than the number of observations of the data set");
  
  if(Kquality >= data.n_rows)
    throw invalid_argument ("The 'Kquality' must be lower than the number of observations of the data set");
  
  if(selectBetterUnitFree && smallerUnitFree <= 0)
    throw invalid_argument ("The 'smallerUnitFree' must be greater than 0 for the 'selectBetterUnitFree' option");
  
  if(selectBetterUnitFree && n_t <= 0)
    throw invalid_argument ("The 'n_t' must be greater than 0 for the 'selectBetterUnitFree' option");
  
  if(selectBetterUnitFree && ratio <= 1)
    throw invalid_argument ("The 'ratio' must be greater than 1 for the 'selectBetterUnitFree' option");
  
  if(applyHiperbolicSmoothing && (rho > 1 || rho < 0))
    throw invalid_argument ("The 'rho' must be greater than 0 and lower than 1");
  
  if(applyHiperbolicSmoothing && gamma == 0)
    throw invalid_argument ("The 'gamma' must be different than 0 for the 'applyHiperbolicSmoothing' option");
  
  if(applyHiperbolicSmoothing && n_gamma <= 0)
    throw invalid_argument ("The 'n_gamma' must be greater than 0 for the 'applyHiperbolicSmoothing' option");
  
  if( Kquality == 0 ) 
    Kquality = Kproj;
  
  
  arma::imat neighborhood = getNeighborhoodMatrix(data, Kproj, true);
  
  arma::imat notNeighborhood = neighborhood;
  notNeighborhood.for_each([](int &val) {
    if (val == 1.0)
      val= 0.0;
    else
      val = 1.0;
  }); 
  
  
  double tau = smallerUnitFree;
  double tt = getParameterT(data , neighborhood, tau);
  
  arma::mat lastConf = conf;
  
  optimResult lastResult;
  
  optimResult bestResult;
  
  LocalContinuityMetaCriterionResult bestLocalContinuityResult;
  LocalContinuityMetaCriterionResult lastLocalContinuityResult;
  
  bestLocalContinuityResult = getLocalContinuityMetaCriterion( data, lastConf, Rn, Kquality);
  
  
  if (selectBetterUnitFree){
    double bestTt=0, bestTau=0,gammaOnTauSelection=0;
    arma::vec tauList =  {};
    arma::vec ttList (n_t);
    
    if(tauList.is_empty())
    {
      tauList.resize(n_t);
      
      for (unsigned int i = 0; i < tauList.n_elem; i++){
        tauList(i) = smallerUnitFree * pow(ratio, i);
        if (i == 0)
          ttList(i) = getParameterT(data , neighborhood, tauList(i));
        else
          ttList(i) = ttList(0) * pow(ratio, i);
      }
      ttList = arma::sort(ttList, "descend");
      tauList = arma::sort(tauList, "descend");
    }
    else{
      for (unsigned int i = 0; i < tauList.n_elem; i++){
        ttList(i) = getParameterT(data , neighborhood, tauList(i));
      }
    }
    
    for(unsigned int i = 0; i < ttList.n_elem; i++){
      if (verbose)
        Rprintf("\nRunning optimLocalMds for tt(%i) = %f and Gamma = 0\n", i,ttList(i));
      
      lastResult = cppOptimHSLocalMds(
        data,
        lastConf,
        neighborhood,
        notNeighborhood,
        Rn,
        ttList(i),
        gammaOnTauSelection,
        maxIt,
        optMethod,
        optTrace,
        optReport
      );
      
      lastConf = lastResult.parameter;
      lastConf.reshape(lastConf.n_cols, lastConf.n_rows);
      lastConf = lastConf.t();
      
      lastLocalContinuityResult = getLocalContinuityMetaCriterion( data, lastConf, Rn, Kquality);
      
      if ( lastLocalContinuityResult.Nk > bestLocalContinuityResult.Nk || i == 0 ){
        bestResult = lastResult;
        bestLocalContinuityResult = lastLocalContinuityResult;
        bestTt = ttList(i);
        bestTau = tauList(i);
      }
      conf = bestResult.parameter - mean( vectorise(bestResult.parameter) );
      
    }
    
    tau = bestTau;
    tt = bestTt;
  }
  
  if (applyHiperbolicSmoothing){
    unsigned int counter= 0;
    
    while ( counter < n_gamma && sqrt(pow(gamma, 2)) > smallerGammaAllowed ){
      
      if (verbose)
        Rprintf("\nRunning optimHSLocalMds for tt = %f and Gamma(%d) = %f\n", tt, counter, gamma);
      
      lastResult = cppOptimHSLocalMds(
        data,
        conf,
        neighborhood,
        notNeighborhood,
        Rn,
        tt,
        gamma,
        maxIt,
        optMethod,
        optTrace,
        optReport
      );
      
      conf = lastResult.parameter;
      
      lastConf = lastResult.parameter;
      lastConf.reshape(lastConf.n_cols, lastConf.n_rows);
      lastConf = lastConf.t();
      lastLocalContinuityResult = getLocalContinuityMetaCriterion( data, lastConf, Rn, Kquality);
      
      if ( lastLocalContinuityResult.Nk > bestLocalContinuityResult.Nk || (counter == 0 && !selectBetterUnitFree ) ){
        bestResult = lastResult;
        bestLocalContinuityResult = lastLocalContinuityResult;
      }
      conf = bestResult.parameter - mean( vectorise(bestResult.parameter) );
      
      gamma = gamma * rho;
      
      counter++;
    }
    
  }

  
  conf = bestResult.parameter;

  double gammaStressTest = 0;
  double stress = getHSLocalMdsStress(data, vectorise(conf), neighborhood,
                                      notNeighborhood, Rn, tt, gammaStressTest);
  
  
  conf = reshape(conf, Rn, data.n_rows);
  conf = conf.t();
  
  HsLocalMdsResult result;
  
  result.LCMC = bestLocalContinuityResult;
  result.opt = bestResult;
  result.conf = conf;
  result.stress = stress;
  result.tau = tau;
  result.tt = tt;
  
  return result;
  
}
