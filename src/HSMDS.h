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

optimResult optimHSMds(arma::mat &data, arma::mat &conf, unsigned int &Rn, double &Gamma, int &maxIt, const std::string &optMethod);

HsMdsResult HSMDS(arma::mat &data,
          arma::mat conf,
          unsigned int Rn = 2,
          unsigned int Kquality = 2,
          bool verbose = false,
          bool applyHiperbolicSmoothing = true,
          double gamma = 1,
          unsigned int n_gamma = 30,
          double rho = 0.5,
          int maxIt = 30,
          const std::string optMethod = "CG");

#endif
