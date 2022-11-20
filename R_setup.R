# Step 1: install R extension for vs code

# Step 2
# install.packages("languageserver")

# Step 3
# sudo apt update
# Run `sudo apt-get install libfribidi-dev`
# libharfbuzz-dev

# Step 4
# install.packages("devtools")

# Step 5
install.packages('Rcpp')
library(devtools)
install_github("wbnicholson/BigVAR/BigVAR")

library("BigVAR")
options(vsc.plot = FALSE)
data(Y)
# Create a Basic VAR-L (Lasso Penalty) with maximum lag order p=4, 10 grid points with lambda optimized according to rolling validation of 1-step ahead MSFE
mod1<-constructModel(Y,p=4,"Basic",gran=c(150,10),h=1,cv="Rolling",verbose=FALSE,IC=TRUE,model.controls=list(intercept=TRUE))
results=cv.BigVAR(mod1)
print(results)
plot(results)