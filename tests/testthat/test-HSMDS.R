
test_that("UShaped HS Local MDS works", {

# A U-shaped curve, (x^4+1), R^2 projected in R^1
set.seed(1)
x<-seq(0.9,1.5,0.05)
xx<-seq(-1,1,0.2)+runif(11,0,0.1)
x<-c(-x,x,xx)
Ccurve<-cbind(x,x^4+1)
d<-stats::dist (Ccurve)
dataset<- as.matrix(d)
Rn = 1


conf = cmdscale(d = dataset, k = Rn)
conf = as.matrix(conf)

RcppHSMDSResult = RcppHSMDS(data = as.matrix(d), 
                            conf = conf,
                            Rn = 1,
                            Kquality = 5,
                            verbose = TRUE,
                            applyHiperbolicSmoothing = TRUE,
                            gamma = mean(d),
                            n_gamma = 1000,
                            rho = 0.5,
                            maxIt = 10000, 
                            optMethod = "CG"
)
titulo <- expression(paste("Blue lines connect the observation in ",
                           R^2," to the projection in ",R^1))
plot(x,x^4+1,ylim=c(-0,6),xlim=c(-6,6),asp=1,
     main="u-shaped curve projected to one dimensional space",
     sub= titulo,cex.sub=0.7,cex.main=0.7)
segments(x,x^4+1,RcppHSMDSResult$conf , rep(0,length(x)), col = "blue")
abline(h=0)


})