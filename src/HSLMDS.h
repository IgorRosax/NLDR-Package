#ifndef HSLocalMDS_H
#define HSLocalMDS_H

#include "HSLMDS_HELPERS.h"
#include <algorithm>
#include <RcppArmadillo.h>
#include <roptim.h>


using namespace std;
using namespace arma;
using namespace roptim;

double getHSLocalMdsStress(arma::mat &data, const arma::vec &confVec, arma::imat &neighborhood, arma::imat &notNeighborhood, unsigned int &Rn, double &tt, double &Gamma);

arma::vec getHSLocalMdsStressGradient(arma::mat &data, const arma::vec &confVec, arma::imat &neighborhood, arma::imat &notNeighborhood, unsigned int &Rn, double &tt, double &Gamma);

optimResult cppOptimHSLocalMds(arma::mat &data, arma::mat &conf, arma::imat &neighborhood, arma::imat &notNeighborhood, unsigned int &Rn, double &tt, double &Gamma, int &maxIt, const std::string &optMethod, unsigned int optTrace, unsigned int optReport);

HsLocalMdsResult cppHSlocalMDS(arma::mat &data,
                               arma::mat &conf,
                               unsigned int Rn,
                               unsigned int Kproj,
                               unsigned int Kquality,
                               bool verbose,
                               bool selectBetterUnitFree,
                               double smallerUnitFree,
                               unsigned int n_t,
                               double ratio,
                               bool applyHyperbolicSmoothing,
                               double gamma,
                               unsigned int n_gamma,
                               double rho,
                               int maxIt,
                               const std::string optMethod,
                               unsigned int optTrace, 
                               unsigned int optReport);

#endif
