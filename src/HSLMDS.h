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

HsLocalMdsResult cppHSlocalMDS( arma::mat &data,
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
                                unsigned int optReport = 10);

#endif
