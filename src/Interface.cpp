// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(roptim)]]
#include "HSLMDS_HELPERS.h"
#include "HSMDS.h"
#include "HSLocalMDS.h"

#include <RcppArmadillo.h>

using namespace std;
using namespace arma;
using namespace Rcpp;

//' Local Continuity Meta Criterion
//' 
//' @param data A distance matrix (dist()) or a square matrix.
//' @param conf The current configuration.
//' @param Rn The dimension of the space which the data will be represented in. The default is Rn=2.
//' @param k The number of neighborhood applied at Local Continuity to assess the quality of the projection.
//' 
//' @return 3 components:
//' @return Nk   Local Continuity Meta Criterion.
//' @return Mk   Local Continuity Meta Criterion Normalizado.
//' @return Mk_adjusted   Local Continuity Meta Criterion Normalizado Ajustado.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List RcppGetLocalContinuityMetaCriterion(arma::mat &data, arma::mat &conf, int Rn, int k){
 LocalContinuityMetaCriterionResult result = getLocalContinuityMetaCriterion (data, conf, Rn, k);
 
 return Rcpp::List::create(
   Rcpp::Named("Nk") = result.Nk,
   Rcpp::Named("Mk") = result.Mk,
   Rcpp::Named("Mk_adjusted") = result.Mk_adjusted
 );
 
}
 
 
 
//' Local Continuity Meta Criterion by Vector
//' 
//' @param data A distance matrix (dist()) or a square matrix.
//' @param conf The current configuration.
//' @param Rn The dimension of the space which the data will be represented in. The default is Rn=2.
//' @param k The number of neighborhood applied at Local Continuity to assess the quality of the projection.
//' 
//' @return 3 components:
//' @return Nk   Local Continuity Meta Criterion.
//' @return Mk   Local Continuity Meta Criterion Normalizado.
//' @return Mk_adjusted   Local Continuity Meta Criterion Normalizado Ajustado.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List RcppGetLocalContinuityMetaCriterionByVector(arma::mat &data, arma::mat &conf, int Rn, int k){
 LocalContinuityMetaCriterionResult result = getLocalContinuityMetaCriterionByVector (data, conf, Rn, k);
 
 return Rcpp::List::create(
   Rcpp::Named("Nk") = result.Nk,
   Rcpp::Named("Mk") = result.Mk,
   Rcpp::Named("Mk_adjusted") = result.Mk_adjusted
 );
 
}
 
 
//' Rcpp Optimization for HS MDS
//' 
//' @param data A distance matrix (dist()) or a square matrix.
//' @param conf An initial configuration.
//' @param Rn The dimension of the space which the data will be represented in. The default is Rn=2.
//' @param Gamma A numeric constant value used for smoothing the configuration.
//' @param maxIt The number of optimizing iterations to assess the quality of the projection.
//' @param optMethod The optimization method. It is possible to choose one of these methods: "Nelder-Mead", "BFGS", "CG", "L-BFGS-B".
//' 
//' @return optimResult   A list with results from optimization process.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List RcppOptimHSMds (arma::mat &data, arma::mat &conf, unsigned int &Rn, double &Gamma, int &maxIt, const std::string &optMethod ){
 optimResult result = optimHSMds (data, conf, Rn, Gamma, maxIt, optMethod);
 
 return Rcpp::List::create(
   Rcpp::Named("parameter") = result.parameter,
   Rcpp::Named("value") = result.value,
   Rcpp::Named("fnCount") = result.fnCount,
   Rcpp::Named("grCount") = result.grCount,
   Rcpp::Named("convergence") = result.convergence,
   Rcpp::Named("message") = result.message
 );
}
 
 
//' Rcpp Optimization for Hyperbolic Smoothing Local MDS
//' 
//' @param data A distance matrix (dist()) or a square matrix.
//' @param conf An initial configuration.
//' @param neighborhood Symmetric matrix that represents which items are neighbors to each other.
//' @param notNeighborhood Symmetric matrix that represents which items are not neighbors to each other.
//' @param Rn The dimension of the space which the data will be represented in. The default is Rn=2.
//' @param tt The repultion.
//' @param Gamma A numeric constant value used for smoothing the configuration.
//' @param maxIt The number of optimizing iterations to assess the quality of the projection.
//' @param optMethod The optimization method. It is possible to choose one of these methods: "Nelder-Mead", "BFGS", "CG", "L-BFGS-B".
//' 
//' @return optimResult   A list with results from optimization process.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List RcppOptimHSLocalMds (arma::mat &data, arma::mat &conf, arma::imat &neighborhood,arma::imat &notNeighborhood, unsigned int &Rn, double &tt, double &Gamma, int &maxIt, const std::string &optMethod ){
 optimResult result = optimHSLocalMds (data, conf, neighborhood, notNeighborhood, Rn, tt, Gamma, maxIt, optMethod);
 
 return Rcpp::List::create(
   Rcpp::Named("parameter") = result.parameter,
   Rcpp::Named("value") = result.value,
   Rcpp::Named("fnCount") = result.fnCount,
   Rcpp::Named("grCount") = result.grCount,
   Rcpp::Named("convergence") = result.convergence,
   Rcpp::Named("message") = result.message
 );
}
 
 
 
//' HS MDS
//'
//' A Smoothing Optimization Approach Applied to the MDS Method
//'
//' @param data A distance matrix (dist()) or a square matrix.
//' @param conf An initial configuration. If none is supplied the configuration will be randomly generated. If x=="cmdscale", the function cmdscale() is used to provide an inicial configuration.
//' @param Rn The dimension of the space which the data will be represented in. The default is Rn=2.
//' @param Kquality The number of neighborhood applied at Local Continuity to assess the quality of the projection. The default is NULL, set the same number at Kproj.
//' @param verbose A boolean constant. If true, informations will be printed on terminal during the execution.
//' @param applyHiperbolicSmoothing A boolean constant. If true, the Hiperbolic Smoothing will be applyed.
//' @param gamma A numeric constant value used for smoothing the configuration.
//' @param n_gamma maximum number of gamma values allowed.
//' @param rho A numeric constant value between 0 and 1 used for decrease gamma value.
//' @param maxIt It is the control argument of the function optim.
//' @param optMethod The optimization method. The default is "CG", a conjugate gradients method.  it is possible to choose one of these methods: "Nelder-Mead", "BFGS", "CG", "L-BFGS-B".
//' @return Five components:
//' @return conf   The points projected in 'd' dimension space.
//' @return LocalContinuityResult   The criterion Local Continuity (LC).
//' @return optimResult   A list with results from optimization process.
//' 
//' @examples
//' # A U-shaped curve, (x^4+1), R^2 projected in R^1
//' set.seed(1)
//' x<-seq(0.9,1.5,0.05)
//' xx<-seq(-1,1,0.2)+runif(11,0,0.1)
//' x<-c(-x,x,xx)
//' Ccurve<-cbind(x,x^4+1)
//' d<-stats::dist (Ccurve)
//' dataset<- as.matrix(d)
//' Rn = 1
//' 
//' 
//' conf = cmdscale(d = dataset, k = Rn)
//' conf = as.matrix(conf)
//' 
//' RcppHSMDSResult = RcppHSMDS(data = as.matrix(d), 
//'                                       conf = conf,
//'                                       Kquality = 5,
//'                                       Rn = 1,
//'                                       maxIt = 10000, 
//'                                       optMethod = "CG"
//' )
//' 
//' titulo <- expression(paste("Blue lines connect the observation in ",
//'                             R^2," to the projection in ",R^1))
//' plot(x,x^4+1,ylim=c(-0,6),xlim=c(-6,6),asp=1,
//'      main="u-shaped curve projected to one dimensional space",
//'      sub= titulo,cex.sub=0.7,cex.main=0.7)
//' segments(x,x^4+1,RcppHSMDSResult$conf , rep(0,length(x)), col = "blue")
//' abline(h=0)
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List RcppHSMDS(arma::mat &data,
                    arma::mat conf,
                    unsigned int Rn = 2,
                    unsigned int Kquality = 2,
                    bool verbose = false,
                    bool applyHiperbolicSmoothing = true,
                    double gamma = 1,
                    unsigned int n_gamma = 30,
                    double rho = 0.5,
                    int maxIt = 30,
                    const std::string optMethod = "CG")
{
 HsMdsResult result;
 
 result = HSMDS(data,
                conf,
                Rn,
                Kquality,
                verbose,
                applyHiperbolicSmoothing,
                gamma,
                n_gamma,
                rho,
                maxIt,
                optMethod);
 
 
 Rcpp::List LcmcList =   Rcpp::List::create(
   Rcpp::Named("Nk") = result.LCMC.Nk,
   Rcpp::Named("Mk") = result.LCMC.Mk,
   Rcpp::Named("Mk_adjusted") = result.LCMC.Mk_adjusted
 );
 
 Rcpp::List optList =   Rcpp::List::create(
   Rcpp::Named("value") = result.opt.value,
   Rcpp::Named("convergence") = result.opt.convergence,
   Rcpp::Named("fnCount") = result.opt.fnCount,
   Rcpp::Named("grCount") = result.opt.grCount,
   Rcpp::Named("message") = result.opt.message
 
 );
 
 return Rcpp::List::create(
   Rcpp::Named("conf") = result.conf,
   Rcpp::Named("stress") = result.stress,
   Rcpp::Named("stressNormalized") = result.stressNormalized,
   Rcpp::Named("LocalContinuityResult") = LcmcList,
   Rcpp::Named("optimResult") = optList
 );
 
}
 
 
 
//' Hyperbolic Smoothing Local MDS
//'
//' A Smoothing Optimization Approach Applied to the Local MDS Method
//'
//' @param data A distance matrix (dist()) or a square matrix.
//' @param conf An initial configuration. If none is supplied the configuration will be randomly generated. If x=="cmdscale", the function cmdscale() is used to provide an inicial configuration.
//' @param Rn The dimension of the space which the data will be represented in. The default is Rn=2.
//' @param Kproj The number of neighborhood to be preserved in the projection.
//' @param Kquality The number of neighborhood applied at Local Continuity to assess the quality of the projection. The default is NULL, set the same number at Kproj.
//' @param verbose A boolean constant. If true, information will be printed on terminal during the execution.
//' @param selectBetterUnitFree A boolean constant. If true, a Geometric sequence will be generated to select the best Unit Free.
//' @param smallerUnitFree the smallest value for the unit free parameter. Geometric sequence's start value.
//' @param n_t length for the grid search.
//' @param ratio ratio of the geometric sequence.
//' @param applyHiperbolicSmoothing A boolean constant. If true, the Hiperbolic Smoothing will be applyed.
//' @param gamma A numeric constant value used for smoothing the configuration.
//' @param n_gamma maximum number of gamma values allowed.
//' @param rho A numeric constant value between 0 and 1 used for decrease gamma value.
//' @param maxIt It is the control argument of the function optim.
//' @param optMethod The optimization method. The default is "CG", a conjugate gradients method.  it is possible to choose one of these methods: "Nelder-Mead", "BFGS", "CG", "L-BFGS-B".
//' @return Five components:
//' @return conf   The points projected in 'd' dimension space.
//' @return LocalContinuityResult   The criterion Local Continuity (LC).
//' @return Tau   The value of the parameter unit free that generate the best projection according to the criterion Local Continuity.
//' @return Tt   The repultion.
//' @return optimResult   A list with results from optimization process.
//' 
//' @references 
//'  Lisha Chen and Andreas Buja, "Local multidimensional scaling for nonlinear dimension reduction, graph drawing, and proximity analysis." Journal of the American Statistical Association, 2009, 104(485), pp.209-219.
//'
//' @examples
//' # A U-shaped curve, (x^4+1), R^2 projected in R^1
//' set.seed(1)
//' x<-seq(0.9,1.5,0.05)
//' xx<-seq(-1,1,0.2)+runif(11,0,0.1)
//' x<-c(-x,x,xx)
//' Ccurve<-cbind(x,x^4+1)
//' d<-stats::dist (Ccurve)
//' dataset<- as.matrix(d)
//' Rn = 1
//' 
//' conf = cmdscale(d = dataset, k = Rn)
//' conf = as.matrix(conf)
//' 
//' RcppHSlocalMDSResult = RcppHSlocalMDS(data = as.matrix(d), 
//'                                       conf = conf,
//'                                       Rn = 1,
//'                                       Kproj = 5, 
//'                                       Kquality = 5,
//'                                       smallerUnitFree = 0.1,
//'                                       applyHiperbolicSmoothing = TRUE,
//'                                       gamma = 1,
//'                                       n_gamma = 10000,
//'                                       rho = (1 / sqrt(10)),
//'                                       maxIt = 10000, 
//'                                       optMethod = "CG"
//' )
//' 
//' titulo <- expression(paste("Blue lines connect the observation in ",
//'                             R^2," to the projection in ",R^1))
//' plot(x,x^4+1,ylim=c(-0,6),xlim=c(-6,6),asp=1,
//'      main="u-shaped curve projected to one dimensional space",
//'      sub= titulo,cex.sub=0.7,cex.main=0.7)
//' segments(x,x^4+1,RcppHSlocalMDSResult$conf , rep(0,length(x)), col = "blue")
//' abline(h=0)
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List RcppHSlocalMDS(arma::mat &data,
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
                         const std::string optMethod = "CG")
{
 HsLocalMdsResult result;
 
 result = HSlocalMDS(data,
                     conf,
                     Rn,
                     Kproj,
                     Kquality,
                     verbose,
                     selectBetterUnitFree,
                     smallerUnitFree,
                     n_t,
                     ratio,
                     applyHiperbolicSmoothing,
                     gamma,
                     n_gamma,
                     rho,
                     maxIt,
                     optMethod);
 
 Rcpp::List LcmcList =   Rcpp::List::create(
   Rcpp::Named("Nk") = result.LCMC.Nk,
   Rcpp::Named("Mk") = result.LCMC.Mk,
   Rcpp::Named("Mk_adjusted") = result.LCMC.Mk_adjusted
 );
 
 Rcpp::List optList =   Rcpp::List::create(
   Rcpp::Named("value") = result.opt.value,
   Rcpp::Named("convergence") = result.opt.convergence,
   Rcpp::Named("fnCount") = result.opt.fnCount,
   Rcpp::Named("grCount") = result.opt.grCount,
   Rcpp::Named("message") = result.opt.message
 
 );
 
 return Rcpp::List::create(
   Rcpp::Named("conf") = result.conf,
   Rcpp::Named("stress") = result.stress,
   Rcpp::Named("LocalContinuityResult") = LcmcList,
   Rcpp::Named("Tau") = result.tau,
   Rcpp::Named("Tt") = result.tt,
   Rcpp::Named("optimResult") = optList
 );
}
