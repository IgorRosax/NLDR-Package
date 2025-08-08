#ifndef HSLMDS_HELPERS_H
#define HSLMDS_HELPERS_H

#include <algorithm>
#include <RcppArmadillo.h>
#include <cmath>

using namespace std;
using namespace arma;


struct optimResult{
  arma::mat parameter;
  double value;
  int fnCount, grCount, convergence;
  string message;
};

struct LocalContinuityMetaCriterionResult {
  double Nk, Mk, Mk_adjusted;
} ;

struct HsMdsResult {
  optimResult opt;
  LocalContinuityMetaCriterionResult LCMC;
  arma::mat conf;
  double stress, stressNormalized;
};

struct HsLocalMdsResult {
  optimResult opt;
  LocalContinuityMetaCriterionResult LCMC;
  arma::mat conf;
  double stress, tau, tt;
};

arma::mat getRankedDistanceMatrix(arma::mat &data);

arma::imat getNeighborhoodMatrix(arma::mat &data, int k, bool symmetric);

arma::ivec getNeighborhoodVector(arma::vec &data, int k);

double getEuclideanDistance(arma::rowvec data1, arma::rowvec data2);

arma::mat getEuclideanDistanceMatrix(arma::mat &data);

arma::mat getEuclideanDistanceMatrix(const arma::mat &data);

arma::vec getEuclideanDistanceVector(arma::mat &data, int i);

LocalContinuityMetaCriterionResult getLocalContinuityMetaCriterion (arma::mat &data, arma::mat &conf, int Rn, int k);

LocalContinuityMetaCriterionResult getLocalContinuityMetaCriterionByVector (arma::mat &data, arma::mat conf, int Rn, int k);

double getParameterT(arma::mat &data, arma::imat &neighborhood, double tau);

arma::mat getHSfTheta(const arma::mat &conf, double &gamma);

double getHSfTheta(double x, double y, double gamma);

#endif
