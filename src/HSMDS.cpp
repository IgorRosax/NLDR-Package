#include "HSLMDS_HELPERS.h"

#include <algorithm>
#include <RcppArmadillo.h>
#include <roptim.h>

using namespace std;
using namespace arma;
using namespace roptim;

//' Get HS MDS Stress
//'
//' This function calculates the Multidimensional Scaling (MDS) stress for the given data.
//'
//' @param data A distance matrix (dist()) or a square matrix.
//' @param confVec A numeric vector representing the configuration vector.
//' @param Rn An integer specifying the number of dimensions for the configuration.
//' @param Gamma A numeric constant value used for smoothing the configuration.
//'
//' @return A numeric value representing the MDS stress.
//'
//' @details
//' The function first reshapes the configuration vector into a matrix and computes the Euclidean distance matrix for the configuration. 
//' It then calculates the difference between the original data and the configuration distance matrix, squares these differences.
//'
//' @examples
//' \dontrun{
//' data <- matrix(runif(100), nrow=10)
//' confVec <- runif(30)
//' Rn <- 3
//' stress <- getHSMdsStress(data, confVec, Rn)
//' }
//'
//' @export
// [[Rcpp::export]]
double getHSMdsStress (arma::mat &data,const arma::vec &confVec, unsigned int &Rn, double &Gamma){
   
   arma::mat confDist;
   
   double Stress=0;
   
   arma::mat conf = reshape(confVec, Rn, data.n_rows);
   conf = conf.t();
   
   confDist = getHSfTheta(conf, Gamma);
   
   for (unsigned int i = 0; i < data.n_rows - 1; i++)
   {
     for (unsigned int j = i + 1; j < data.n_rows; j++)
     {
       Stress += pow( data(i, j) - confDist(i, j), 2 );
     }
   }
   
   return( Stress );
 }
 
 
//' Get HS MDS Stress Normalized
//'
//' This function calculates the Multidimensional Scaling (MDS) stress for the given data.
//'
//' @param data A distance matrix (dist()) or a square matrix.
//' @param confVec A numeric vector representing the configuration vector.
//' @param Rn An integer specifying the number of dimensions for the configuration.
//' @param Gamma A numeric constant value used for smoothing the configuration.
//'
//' @return A numeric value representing the MDS stress.
//'
//' @details
//' The function first reshapes the configuration vector into a matrix and computes the Euclidean distance matrix for the configuration. 
//' It then calculates the difference between the original data and the configuration distance matrix, squares these differences and divides by the original data distance squares sum.
//'
//' @examples
//' \dontrun{
//' data <- matrix(runif(100), nrow=10)
//' confVec <- runif(30)
//' Rn <- 3
//' stress <- getHSMdsStressNormalized(data, confVec, Rn)
//' }
//'
//' @export
// [[Rcpp::export]]
double getHSMdsStressNormalized (arma::mat &data,const arma::vec &confVec, unsigned int &Rn, double &Gamma){
   
   arma::mat confDist;
   
   double Stress = 0, squareData = 0;
   
   arma::mat conf = reshape(confVec, Rn, data.n_rows);
   conf = conf.t();
   
   confDist = getHSfTheta(conf, Gamma);
   
   for (unsigned int i = 0; i < data.n_rows - 1; i++)
   {
     for (unsigned int j = i + 1; j < data.n_rows; j++)
     {
       Stress += pow( data(i, j) - confDist(i, j), 2 );
       squareData += pow(data(i, j) ,2);
     }
   }
   
   return( Stress / squareData );
 }
 
//' Get HS MDS Stress Gradient
//'
//' This function calculates the gradient of the HS Multidimensional Scaling (MDS) stress for the given data.
//'
//' @param data A distance matrix (dist()) or a square matrix.
//' @param confVec A numeric vector representing the configuration vector.
//' @param Rn An integer specifying the number of dimensions for the configuration.
//' @param Gamma A numeric constant value used for smoothing the configuration.
//'
//' @return A numeric vector representing the gradient of the local MDS stress.
//'
//' @details
//' The function reshapes the configuration vector into a matrix and computes the Euclidean distance matrix for the configuration. 
//' It then calculates the difference between the original data and the configuration distance matrix, normalizes these differences, 
//' and computes the gradient. The final gradient is returned as a vector.
//'
//' @examples
//' \dontrun{
//' data <- matrix(runif(100), nrow=10)
//' confVec <- runif(30)
//' Rn <- 3
//' gradient <- getHSMdsStressGradient(data, confVec, Rn)
//' }
//'
//' @export
// [[Rcpp::export]]
arma::vec getHSMdsStressGradient(arma::mat &data,const arma::vec &confVec, unsigned int &Rn, double &Gamma){
   
   arma::mat confDist;
   arma::mat distDif;
   arma::mat parcela;
   arma::mat grad(data.n_rows, Rn, arma::fill::zeros);
   arma::mat conf = reshape(confVec, Rn, data.n_rows);
   
   //conf.reshape(conf.n_cols, conf.n_rows);
   conf = conf.t();
   
   confDist = getHSfTheta(conf, Gamma);
   
   distDif = (data - confDist);
   
   for(unsigned int i = 0; i < data.n_rows - 1; i++){
     for(unsigned int j = i + 1; j < data.n_rows; j++){
       grad.row(i) = grad.row(i) + ((+2) * distDif(i,j) * (conf.row(i) - conf.row(j)) / confDist(i,j));
       grad.row(j) = grad.row(j) + ((+2) * distDif(j,i) * (conf.row(j) - conf.row(i)) / confDist(j,i));
     }
   }
   return vectorise(-1*(grad.t())); 
 }
 
 class optimHSMdsStress : public Functor{
 public:
   optimHSMdsStress(
     arma::mat &data,
     arma::mat &conf
   ) : data(data), conf(conf) {}
   
   double operator()(const arma::vec &x ) override {
     return getHSMdsStress(data, x, Rn, Gamma);
   }
   
   void Gradient(const arma::vec &x, arma::vec &grad) override {
     grad = getHSMdsStressGradient(data, x, Rn, Gamma);
     
   }
   arma::mat getData (){return data;}
   arma::mat getConf (){return conf;}
   int getRn () {return Rn;}
   
   void setData (arma::mat &data){this->data = data;}
   void setConf (arma::mat &conf){this->conf = conf;}
   void setRn (unsigned int Rn) {this->Rn = Rn;}
   void setGamma (double Gamma) {this->Gamma = Gamma;}
 private:
   arma::mat &data;
   arma::mat &conf;
   unsigned int Rn = 0;
   double Gamma = 0;
 };
 
 
 optimResult cppOptimHSMds (arma::mat &data, arma::mat &conf, unsigned int &Rn, double &Gamma, 
                         int &maxIt, const std::string &optMethod, unsigned int trace, unsigned int optReport){
   optimHSMdsStress optimStress (data, conf);
   Roptim<optimHSMdsStress> opt(optMethod);
   optimResult result;
   
   optimStress.setRn(Rn);
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

 HsMdsResult cppHSMDS(arma::mat &data,
                   arma::mat conf,
                   unsigned int Rn = 2,
                   unsigned int Kquality = 2,
                   bool verbose = false,
                   bool applyHiperbolicSmoothing = true,
                   double gamma = 1,
                   unsigned int n_gamma = 30,
                   double rho = 0.5,
                   int maxIt = 30,
                   const std::string optMethod = "CG",
                   unsigned int optTrace = 0, 
                   unsigned int optReport = 10){
   
   if( !data.is_square() )
     throw invalid_argument("distances must be result of 'dist' or a square matrix");
   
   if ( conf.n_rows!=data.n_rows || conf.n_cols!=Rn ) 
     throw invalid_argument("The number of rows of the 'conf' must be the same of the numbers of observations of the data set. The number of columns must be the same of the 'Rn'");
   
   if(Kquality >= data.n_rows)
     throw invalid_argument ("The 'Kquality' must be lower than the number of observations of the data set");
   
   if(applyHiperbolicSmoothing && (rho > 1 || rho < 0))
     throw invalid_argument ("The 'rho' must be greater than 0 and lower than 1");
   
   if(applyHiperbolicSmoothing && gamma == 0)
     throw invalid_argument ("The 'gamma' must be different than 0 for the 'applyHiperbolicSmoothing' option");
   
   if(applyHiperbolicSmoothing && n_gamma <= 0)
     throw invalid_argument ("The 'n_gamma' must be greater than 0 for the 'applyHiperbolicSmoothing' option");
   
   optimResult lastResult;
   optimResult bestResult;
   
   if (applyHiperbolicSmoothing){
     
     unsigned int counter= 0;
     
     while ( counter < n_gamma && sqrt(pow(gamma, 2)) > pow(10, -16) ){
       if(verbose)
         Rprintf("\nRunning optimHSMds for Gamma(%d) = %f\n", counter,gamma);
       lastResult = cppOptimHSMds(
         data,
         conf,
         Rn,
         gamma,
         maxIt,
         optMethod,
         optTrace,
         optReport
       );
       
       conf = lastResult.parameter;
       
       
       if ( counter == 0  || lastResult.value < bestResult.value ){
         bestResult = lastResult;
       }
       
       gamma = gamma * rho;
       
       counter++;
     }
   }else{
     double gammaWithoutHS = 0;
     if(verbose)
       Rprintf("\nRunning optimHSMds for Gamma = 0\n");
     bestResult = cppOptimHSMds(
       data,
       conf,
       Rn,
       gammaWithoutHS,
       maxIt,
       optMethod,
       optTrace,
       optReport
     );
     
     conf = lastResult.parameter;
   }
   
   
   LocalContinuityMetaCriterionResult LocalContinuityResult;
   
   conf = bestResult.parameter - mean( vectorise(bestResult.parameter) );
   
   double gammaStressTest = 0;
   double stress = getHSMdsStress(data, vectorise(conf), Rn, gammaStressTest);
   
   double stressNormalized = getHSMdsStressNormalized(data, vectorise(conf), Rn, gammaStressTest);
   
   conf = reshape(conf, Rn, data.n_rows);
   conf = conf.t();
   
   LocalContinuityResult = getLocalContinuityMetaCriterion( data, conf, Rn, Kquality);
   
   HsMdsResult result;
   result.LCMC = LocalContinuityResult;
   result.opt = bestResult;
   result.conf = conf;
   result.stress = stress;
   result.stressNormalized = stressNormalized;
   
   return result;
 }
 
