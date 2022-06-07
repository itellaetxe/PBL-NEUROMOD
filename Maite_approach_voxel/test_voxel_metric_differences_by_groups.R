# Clear workspace
rm(list = ls())

getwd()
setwd("C:/Users/Iñigo/PycharmProjects/PBL-NEUROMOD/Maite_approach_voxel/")

#### Get CN and AD subject names ####
cndirs <- basename(list.dirs("E:/PBL_MASTER/CN_CONNMAT"))
cndirs <- cndirs[2:length(cndirs)]
cntestdirs <- basename(list.dirs("E:/PBL_MASTER/TEST/NC"))
cntestdirs <- cntestdirs[2:length(cntestdirs)]
cnsubjs <- c(cndirs, cntestdirs)


addirs <- basename(list.dirs("E:/PBL_MASTER/AD_CONNMAT"))
addirs <- addirs[2:length(addirs)]
adtestdirs <- basename(list.dirs("E:/PBL_MASTER/TEST/AD"))
adtestdirs <- adtestdirs[2:length(adtestdirs)]
adsubjs <- c(addirs, adtestdirs)


# Load and make dataframe out of FA and MD
dirs <- list.dirs("E:/PBL_MASTER/VOXEL_METRICS")
dirs <- dirs[2:length(dirs)]

#### FA (paired regions) ####
folder <- dirs[1]
subname <- basename(folder)
data_path <- paste(folder, list.files(folder, pattern = "*region_pairs_fa_means.csv"), sep="/")
datuk <- read.csv(data_path, sep=",", header=TRUE)
FA_dual_region_df <- data.frame(matrix(nrow=length(cnsubjs)+length(adsubjs),
                                       ncol=length(datuk$Parcel)+1))
colnames(FA_dual_region_df) <- c(datuk$Parcel, "Label")

i <- 1
for (folder in dirs) {
  subname <- basename(folder)
  data_path <- paste(folder, list.files(folder, pattern = "*region_pairs_fa_means.csv"), sep="/")
  datuk <- read.csv(data_path, sep=",", header=TRUE)
  FA_dual_region_df[i, 1:(ncol(FA_dual_region_df)-1)] <- datuk$FA_mean
  if (subname %in% cnsubjs) {
    FA_dual_region_df[i, ncol(FA_dual_region_df)] <- 0
  }
  else {
    FA_dual_region_df[i, ncol(FA_dual_region_df)] <- 1
  }
  i <- i + 1
}

# Remove row 37 because it is incorrect (lots of NA)
FA_dual_region_df <- FA_dual_region_df[-37,]
cat("Conflictive, incorrect subject is: ", basename(dirs[37]), sep="")

regions <- datuk$Parcel
i <- 1
pvals_fa_dual <- NULL
for (region in regions) {
  ad <- FA_dual_region_df[[region]][FA_dual_region_df$Label==1]
  cn <- FA_dual_region_df[[region]][FA_dual_region_df$Label==0]
  vt <- var.test(ad, cn)
  if (vt$p.value < 0.05) {
    varbool <- FALSE
  }
  else{
    varbool <- TRUE
  }
  tt <- t.test(ad, cn, var.equal=varbool)
  pvals_fa_dual <- c(pvals_fa_dual,tt$p.value)
  i <- i + 1
}

fdr_corrected_fa_dual <- p.adjust(pvals_fa_dual, method="fdr", n=length(pvals_fa_dual))
bonf_corrected_fa_dual <- p.adjust(pvals_fa_dual, method="bonferroni", n=length(pvals_fa_dual))

regions[fdr_corrected_fa_dual<0.05]

#### MD (paired regions) ####
folder <- dirs[1]
subname <- basename(folder)
data_path <- paste(folder, list.files(folder, pattern = "*region_pairs_md_means.csv"), sep="/")
datuk <- read.csv(data_path, sep=",", header=TRUE)
MD_dual_region_df <- data.frame(matrix(nrow=length(cnsubjs)+length(adsubjs),
                                       ncol=length(datuk$Parcel)+1))
colnames(MD_dual_region_df) <- c(datuk$Parcel, "Label")

i <- 1
for (folder in dirs) {
  subname <- basename(folder)
  data_path <- paste(folder, list.files(folder, pattern = "*region_pairs_md_means.csv"), sep="/")
  datuk <- read.csv(data_path, sep=",", header=TRUE)
  MD_dual_region_df[i, 1:(ncol(MD_dual_region_df)-1)] <- datuk$MD_mean
  if (subname %in% cnsubjs) {
    MD_dual_region_df[i, ncol(MD_dual_region_df)] <- 0
  }
  else {
    MD_dual_region_df[i, ncol(MD_dual_region_df)] <- 1
  }
  i <- i + 1
}

# Remove row 37 because it is incorrect (lots of NA)
MD_dual_region_df <- MD_dual_region_df[-37,]
cat("Conflictive, incorrect subject is: ", basename(dirs[37]), sep="")

regions <- datuk$Parcel
i <- 1
pvals_md_dual <- NULL
for (region in regions) {
  ad <- MD_dual_region_df[[region]][MD_dual_region_df$Label==1]
  cn <- MD_dual_region_df[[region]][MD_dual_region_df$Label==0]
  vt <- var.test(ad, cn)
  if (vt$p.value < 0.05) {
    varbool <- FALSE
  }
  else{
    varbool <- TRUE
  }
  tt <- t.test(ad, cn, var.equal=varbool)
  pvals_md_dual <- c(pvals_md_dual,tt$p.value)
  i <- i + 1
}

fdr_corrected_md_dual <- p.adjust(pvals_md_dual, method="fdr", n=length(pvals_md_dual))
bonf_corrected_md_dual <- p.adjust(pvals_md_dual, method="bonferroni", n=length(pvals_md_dual))

regions[fdr_corrected_md_dual<0.05]


#### FA (individual regions) ####
folder <- dirs[1]
subname <- basename(folder)
data_path <- paste(folder, list.files(folder, pattern = "*regional_fa_means.csv"), sep="/")
datuk <- read.csv(data_path, sep=",", header=TRUE)
FA_region_df <- data.frame(matrix(nrow=length(cnsubjs)+length(adsubjs),
                                       ncol=length(datuk$Parcel)+1))
colnames(FA_region_df) <- c(datuk$Parcel, "Label")

i <- 1
for (folder in dirs) {
  subname <- basename(folder)
  data_path <- paste(folder, list.files(folder, pattern = "*regional_fa_means.csv"), sep="/")
  datuk <- read.csv(data_path, sep=",", header=TRUE)
  FA_region_df[i, 1:(ncol(FA_region_df)-1)] <- datuk$FA_mean
  if (subname %in% cnsubjs) {
    FA_region_df[i, ncol(FA_region_df)] <- 0
  }
  else {
    FA_region_df[i, ncol(FA_region_df)] <- 1
  }
  i <- i + 1
}

# Remove row 37 because it is incorrect (lots of NA)
FA_region_df <- FA_region_df[-37,]
cat("Conflictive, incorrect subject is: ", basename(dirs[37]), sep="")

regions <- datuk$Parcel
i <- 1
pvals_fa_indiv <- NULL
for (region in regions) {
  ad <- FA_region_df[[region]][FA_region_df$Label==1]
  cn <- FA_region_df[[region]][FA_region_df$Label==0]
  vt <- var.test(ad, cn)
  if (vt$p.value < 0.05) {
    varbool <- FALSE
  }
  else{
    varbool <- TRUE
  }
  tt <- t.test(ad, cn, var.equal=varbool)
  pvals_fa_indiv <- c(pvals_fa_indiv,tt$p.value)
  i <- i + 1
}

fdr_corrected_fa_indiv <- p.adjust(pvals_fa_indiv, method="fdr", n=length(pvals_fa_indiv))
bonf_corrected_fa_indiv <- p.adjust(pvals_fa_indiv, method="bonferroni", n=length(pvals_fa_indiv))

regions[fdr_corrected_fa_indiv<0.05]
regions[bonf_corrected_fa_indiv<0.05]


#### MD (individual regions) ####
folder <- dirs[1]
subname <- basename(folder)
data_path <- paste(folder, list.files(folder, pattern = "*regional_md_means.csv"), sep="/")
datuk <- read.csv(data_path, sep=",", header=TRUE)
MD_region_df <- data.frame(matrix(nrow=length(cnsubjs)+length(adsubjs),
                                  ncol=length(datuk$Parcel)+1))
colnames(MD_region_df) <- c(datuk$Parcel, "Label")

i <- 1
for (folder in dirs) {
  subname <- basename(folder)
  data_path <- paste(folder, list.files(folder, pattern = "*regional_md_means.csv"), sep="/")
  datuk <- read.csv(data_path, sep=",", header=TRUE)
  MD_region_df[i, 1:(ncol(MD_region_df)-1)] <- datuk$MD_mean
  if (subname %in% cnsubjs) {
    MD_region_df[i, ncol(MD_region_df)] <- 0
  }
  else {
    MD_region_df[i, ncol(MD_region_df)] <- 1
  }
  i <- i + 1
}

# Remove row 37 because it is incorrect (lots of NA)
MD_region_df <- MD_region_df[-37,]
cat("Conflictive, incorrect subject is: ", basename(dirs[37]), sep="")

regions <- datuk$Parcel
i <- 1
pvals_md_indiv <- NULL
for (region in regions) {
  ad <- MD_region_df[[region]][MD_region_df$Label==1]
  cn <- MD_region_df[[region]][MD_region_df$Label==0]
  vt <- var.test(ad, cn)
  if (vt$p.value < 0.05) {
    varbool <- FALSE
  }
  else{
    varbool <- TRUE
  }
  tt <- t.test(ad, cn, var.equal=varbool)
  pvals_md_indiv <- c(pvals_md_indiv,tt$p.value)
  i <- i + 1
}

fdr_corrected_md_indiv <- p.adjust(pvals_md_indiv, method="fdr", n=length(pvals_md_indiv))
bonf_corrected_md_indiv <- p.adjust(pvals_md_indiv, method="bonferroni", n=length(pvals_md_indiv))

regions[fdr_corrected_md_indiv<0.05]
regions[bonf_corrected_md_indiv<0.05]

