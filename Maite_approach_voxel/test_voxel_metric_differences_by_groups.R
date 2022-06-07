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
dirs <- list.dirs("E:/PBL_MASTER/done_VOXEL_METRICS")
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

zatitxu <- FA_dual_region_df[1:29, ]

regions <- datuk$Parcel
i <- 1
for (region in regions) {
  ad <- zatitxu[[regions[i]]][zatitxu$Label==1]
  cn <- zatitxu[[regions[i]]][zatitxu$Label==0]
  vt <- var.test(ad, cn)
  if (vt$p.value < 0.05/length(regions)) {
    varbool <- FALSE
  }
  else{
    varbool <- TRUE
  }
  tt <- t.test(ad, cn, var.equal=varbool)
  pvals_fa_dual[i] <- tt$p.value
  i <- i + 1
}

#### MD (paired regions) ####



#### FA (individual regions) ####



#### MD (individual regions) ####
