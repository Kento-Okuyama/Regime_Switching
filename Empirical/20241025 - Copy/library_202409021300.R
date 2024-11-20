library_load <- function() {
  
  # List of packages to install
  packages_to_install <- c('reshape', 'sigmoid', 'Rlab', 'Rmpfr', 'lavaan', 'mice', 'ranger', 'reshape2',
                           'torch', 'reticulate', 'data.table', 'dplyr', 'tidyverse', 'abind')
  
  # Function to check if a package is installed
  is_package_installed <- function(package_name) {
    return(package_name %in% installed.packages()[, "Package"]) }
  
  # Loop through the list of packages and install them if not installed
  for (package in packages_to_install) {
    if (!is_package_installed(package)) {
      install.packages(package, dependencies = TRUE) } }
  
  library(reshape)
  library(sigmoid)
  library(Rlab)
  library(Rmpfr)
  library(lavaan)
  library(mice)
  library(ranger)
  library(reshape2)
  library(torch)
  library(reticulate)
  library(data.table)
  library(dplyr)
  library(tidyverse)
  library(abind)
  
}