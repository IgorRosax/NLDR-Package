#ifndef HSMDS_H
#define HSMDS_H

#include "HSLMDS_HELPERS.h"

#include <algorithm>
#include <RcppArmadillo.h>
#include <roptim.h>

using namespace std;
using namespace arma;
using namespace roptim;


double getHSMdsStress(arma::mat &data,const arma::vec &confVec, unsigned int &Rn, double &Gamma);

double getHSMdsStressNormalized(arma::mat &data, const arma::vec &confVec, unsigned int &Rn, double &Gamma);
  
arma::mat getHSMdsStressGradient(arma::mat &data, const arma::vec &confVec, unsigned int Rn, double Gamma);

optimResult cppOptimHSMds(arma::mat &data, arma::mat &conf, unsigned int &Rn, double &Gamma, int &maxIt, const std::string &optMethod, unsigned int optTrace, unsigned int optReport);

HsMdsResult cppHSMDS(arma::mat &data,
                     arma::mat &conf,
                     unsigned int Rn,
                     unsigned int Kquality,
                     bool verbose,
                     bool applyHyperbolicSmoothing,
                     double gamma,
                     unsigned int n_gamma,
                     double rho,
                     int maxIt,
                     const std::string optMethod,
                     unsigned int optTrace, 
                     unsigned int optReport);

#endif
