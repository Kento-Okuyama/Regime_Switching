library_load <- function() {
  
  # List of packages to install
  packages_to_install <- c('reshape', 'ggplot2', 'plotly', 'sigmoid', 'Rlab', 'Rmpfr', 'cowplot', 'lavaan', 
                           'torch', 'reticulate', 'cowplot', 'data.table', 'dplyr', 'tidyverse', 'RColorBrewer', 'abind')
  
  # Function to check if a package is installed
  is_package_installed <- function(package_name) {
    return(package_name %in% installed.packages()[, "Package"]) }
  
  # Loop through the list of packages and install them if not installed
  for (package in packages_to_install) {
    if (!is_package_installed(package)) {
      install.packages(package, dependencies = TRUE) } }
  
  library(reshape)
  library(ggplot2)
  library(plotly)
  library(sigmoid)
  library(Rlab)
  library(Rmpfr)
  library(cowplot)
  library(lavaan)
  library(torch)
  library(reticulate)
  library(cowplot)
  library(data.table)
  library(dplyr)
  library(tidyverse)
  library(RColorBrewer)
  library(abind)

}