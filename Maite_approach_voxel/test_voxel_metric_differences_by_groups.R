# Clear workspace
rm(list = ls())
set.seed(7)
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

### INDICES TO DROP ###
# idxs_to_drop <- c(1,3,5,17,9,21,13,26,17,39,31, 94, 92, 90, 88, 72, 68, 78, 80)


# Load and make dataframe out of FA and MD
dirs <- list.dirs("E:/PBL_MASTER/VOXEL_METRICS")
dirs <- dirs[2:length(dirs)]



#### FA (paired regions) ####
# Create empty dataframe with colnames
folder <- dirs[1]
subname <- basename(folder)
data_path <- paste(folder, list.files(folder, pattern = "*region_pairs_fa_means.csv"), sep="/")
datuk <- read.csv(data_path, sep=",", header=TRUE)
FA_dual_region_df <- data.frame(matrix(nrow=length(cnsubjs)+length(adsubjs),
                                       ncol=length(datuk$Parcel)+1))
colnames(FA_dual_region_df) <- c(datuk$Parcel, "Label")
# Fill dataframe
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
# FA_dual_region_test <- FA_dual_region_df[idxs_to_drop,]
# write.table(FA_dual_region_test, file="FA_dual_test.txt", sep=",", row.names=FALSE)
# FA_dual_region_df <- FA_dual_region_df[-idxs_to_drop,]


regions <- datuk$Parcel
nsubs <- nrow(FA_dual_region_df)

all_tests_FA_dual <- matrix(nrow=nsubs, ncol=length(regions))
# Loop thru all subjects leaving one at a time for L1O feature selection.
for (subj in c(1:nsubs)) {
  L1O_df <- FA_dual_region_df[-subj,]
  pvals_fa_dual <- NULL
  for (region in regions) {
    ad <- L1O_df[[region]][L1O_df$Label==1]
    cn <- L1O_df[[region]][L1O_df$Label==0]
    vt <- var.test(ad, cn)
    if (vt$p.value < 0.05) {
      varbool <- FALSE
    } else {
      varbool <- TRUE
    }
    tt <- t.test(ad, cn, var.equal=varbool)
    pvals_fa_dual <- c(pvals_fa_dual, tt$p.value)
  }
  fdr_corrected_fa_dual <- p.adjust(pvals_fa_dual, method="fdr", n=length(pvals_fa_dual))
  all_tests_FA_dual[subj, ] <- fdr_corrected_fa_dual
}
truthness_mat <- all_tests_FA_dual<0.05

regstowrite_fa_dual <- regions[colSums(truthness_mat) == nrow(truthness_mat)]
write.table(regstowrite_fa_dual, file="FA_dual_regs_FDR_0p05_L1O.txt", col.names=FALSE, row.names=FALSE)
write.table(FA_dual_region_df, file="FA_dual_train_L1O.txt", sep=",", row.names=FALSE)




#### MD (paired regions) ####
# Create empty dataframe with colnames
folder <- dirs[1]
subname <- basename(folder)
data_path <- paste(folder, list.files(folder, pattern = "*region_pairs_md_means.csv"), sep="/")
datuk <- read.csv(data_path, sep=",", header=TRUE)
MD_dual_region_df <- data.frame(matrix(nrow=length(cnsubjs)+length(adsubjs),
                                       ncol=length(datuk$Parcel)+1))
colnames(MD_dual_region_df) <- c(datuk$Parcel, "Label")
# Fill dataframe
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
# MD_dual_region_test <- MD_dual_region_df[idxs_to_drop,]
# write.table(MD_dual_region_test, file="MD_dual_test.txt", sep=",", row.names=FALSE)
# MD_dual_region_df <- MD_dual_region_df[-idxs_to_drop,]

regions <- datuk$Parcel
nsubs <- nrow(MD_dual_region_df)

all_tests_MD_dual <- matrix(nrow=nsubs, ncol=length(regions))
# Loop thru all subjects leaving one at a time for L1O feature selection.
for (subj in c(1:nsubs)) {
  L1O_df <- MD_dual_region_df[-subj,]
  pvals_md_dual <- NULL
  for (region in regions) {
    ad <- L1O_df[[region]][L1O_df$Label==1]
    cn <- L1O_df[[region]][L1O_df$Label==0]
    vt <- var.test(ad, cn)
    if (vt$p.value < 0.05) {
      varbool <- FALSE
    } else {
      varbool <- TRUE
    }
    tt <- t.test(ad, cn, var.equal=varbool)
    pvals_md_dual <- c(pvals_md_dual, tt$p.value)
  }
  fdr_corrected_md_dual <- p.adjust(pvals_md_dual, method="fdr", n=length(pvals_md_dual))
  all_tests_MD_dual[subj, ] <- fdr_corrected_md_dual
}
truthness_mat <- all_tests_MD_dual<0.05

regstowrite_md_dual <- regions[colSums(truthness_mat) == nrow(truthness_mat)]
write.table(regstowrite_md_dual, file="MD_dual_regs_FDR_0p05_L1O.txt", col.names=FALSE, row.names=FALSE)
write.table(MD_dual_region_df, file="MD_dual_train_L1O.txt", sep=",", row.names=FALSE)



#### FA (individual regions) ####
# Create empty dataframe with colnames
folder <- dirs[1]
subname <- basename(folder)
data_path <- paste(folder, list.files(folder, pattern = "*regional_fa_means.csv"), sep="/")
datuk <- read.csv(data_path, sep=",", header=TRUE)
FA_region_df <- data.frame(matrix(nrow=length(cnsubjs)+length(adsubjs),
                                  ncol=length(datuk$Parcel)+1))
colnames(FA_region_df) <- c(datuk$Parcel, "Label")
# Fill dataframe
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
# FA_region_test <- FA_region_df[idxs_to_drop,]
# write.table(FA_region_test, file="FA_indiv_test.txt", sep=",", row.names=FALSE)
# FA_region_df <- FA_region_df[-idxs_to_drop,]

regions <- datuk$Parcel
nsubs <- nrow(FA_region_df)

all_tests_FA_indiv <- matrix(nrow=nsubs, ncol=length(regions))
# Loop thru all subjects leaving one at a time for L1O feature selection.
for (subj in c(1:nsubs)) {
  L1O_df <- FA_region_df[-subj,]
  pvals_fa_indiv <- NULL
  for (region in regions) {
    ad <- L1O_df[[region]][L1O_df$Label==1]
    cn <- L1O_df[[region]][L1O_df$Label==0]
    vt <- var.test(ad, cn)
    if (vt$p.value < 0.05) {
      varbool <- FALSE
    } else {
      varbool <- TRUE
    }
    tt <- t.test(ad, cn, var.equal=varbool)
    pvals_fa_indiv <- c(pvals_fa_indiv, tt$p.value)
  }
  fdr_corrected_fa <- p.adjust(pvals_fa_indiv, method="fdr", n=length(pvals_fa_indiv))
  all_tests_FA_indiv[subj, ] <- fdr_corrected_fa
}
truthness_mat <- all_tests_FA_indiv<0.05

regstowrite_fa_indiv <- regions[colSums(truthness_mat) == nrow(truthness_mat)]
write.table(regstowrite_fa_indiv, file="FA_indiv_regs_FDR_0p05_L1O.txt", col.names=FALSE, row.names=FALSE)
write.table(FA_region_df, file="FA_indiv_train_L1O.txt", sep=",", row.names=FALSE)




#### MD (individual regions) ####
# Create empty dataframe with colnames
folder <- dirs[1]
subname <- basename(folder)
data_path <- paste(folder, list.files(folder, pattern = "*regional_md_means.csv"), sep="/")
datuk <- read.csv(data_path, sep=",", header=TRUE)
MD_region_df <- data.frame(matrix(nrow=length(cnsubjs)+length(adsubjs),
                                  ncol=length(datuk$Parcel)+1))
colnames(MD_region_df) <- c(datuk$Parcel, "Label")
# Fill dataframe
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
# MD_region_test <- MD_region_df[idxs_to_drop,]
# write.table(MD_region_test, file="MD_indiv_test.txt", sep=",", row.names=FALSE)
# MD_region_df <- MD_region_df[-idxs_to_drop,]

regions <- datuk$Parcel
nsubs <- nrow(MD_region_df)

all_tests_MD_indiv <- matrix(nrow=nsubs, ncol=length(regions))
# Loop thru all subjects leaving one at a time for L1O feature selection.
for (subj in c(1:nsubs)) {
  L1O_df <- MD_region_df[-subj,]
  pvals_md_indiv <- NULL
  for (region in regions) {
    ad <- L1O_df[[region]][L1O_df$Label==1]
    cn <- L1O_df[[region]][L1O_df$Label==0]
    vt <- var.test(ad, cn)
    if (vt$p.value < 0.05) {
      varbool <- FALSE
    } else {
      varbool <- TRUE
    }
    tt <- t.test(ad, cn, var.equal=varbool)
    pvals_md_indiv <- c(pvals_md_indiv, tt$p.value)
  }
  fdr_corrected_md <- p.adjust(pvals_md_indiv, method="fdr", n=length(pvals_md_indiv))
  all_tests_MD_indiv[subj, ] <- fdr_corrected_md
}
truthness_mat <- all_tests_MD_indiv<0.05

regstowrite_md_indiv <- regions[colSums(truthness_mat) == nrow(truthness_mat)]
write.table(regstowrite_md_indiv, file="MD_indiv_regs_FDR_0p05_L1O.txt", col.names=FALSE, row.names=FALSE)
write.table(MD_region_df, file="MD_indiv_train_L10.txt", sep=",", row.names=FALSE)



#### PCA SECTION ####
library(FactoMineR)
library(factoextra)
library(ggplot2)


#### PCA OF FA DUAL, FILTERED ####
FA_dual_filtered <- FA_dual_region_df[, regstowrite_fa_dual]
FA_dual_filtered_pca <- PCA(FA_dual_filtered, graph=FALSE, scale.unit=TRUE, ncp=ncol(FA_dual_filtered))

par(mfrow=c(2,2))
plot(main="Scree plot: FA dual, filtered", xlab="PC", ylab="Explained Variance %",
     x=c(1:ncol(FA_dual_filtered)), y=FA_dual_filtered_pca$eig[,2], col="blue",
     pch=20, cex=2, xaxt="n", yaxt="n", cex.axis=1, cex.main=1.5)
axis(1, at=seq(1, length(FA_dual_filtered_pca$eig[,2]), 1), cex.lab=1.2, cex.axis=1.2)
axis(2, at=seq(0, 60, 20), cex.axis=1.2)
axis(2, at=c(10, 30, 50), cex.axis=0.9)


#### PCA OF FA INDIVIDUAL, FILTERED ####
FA_indiv_filtered <- FA_region_df[, regstowrite_fa_indiv]
FA_indiv_filtered_pca <- PCA(FA_indiv_filtered, graph=FALSE, scale.unit=TRUE, ncp=ncol(FA_indiv_filtered))

plot(main="Scree plot: FA indiv., filtered", xlab="PC", ylab="Explained Variance %",
     x=c(1:ncol(FA_indiv_filtered)), y=FA_indiv_filtered_pca$eig[,2], col="blue",
     pch=20, cex=2, xaxt="n", yaxt="n", cex.axis=1, cex.main=1.5)
axis(1, at=seq(1, length(FA_indiv_filtered_pca$eig[,2]), 1), cex.lab=1.2, cex.axis=1.2)
axis(2, at=seq(0, 60, 20), cex.axis=1.2)
axis(2, at=c(10, 30, 48), cex.axis=0.9)



#### PCA OF MD DUAL, FILTERED ####
MD_dual_filtered <- MD_dual_region_df[, regstowrite_md_dual]
MD_dual_filtered_pca <- PCA(MD_dual_filtered, graph=FALSE, scale.unit=TRUE, ncp=ncol(MD_dual_filtered))

plot(main="Scree plot: MD dual, filtered", xlab="PC", ylab="Explained Variance %",
     x=c(1:nrow(MD_dual_filtered_pca$eig)), y=MD_dual_filtered_pca$eig[,2], col="red",
     pch=20, cex=2, xaxt="n", yaxt="n", cex.axis=1, cex.main=1.5)
axis(1, at=c(1, seq(5, length(MD_dual_filtered_pca$eig[,2]), 5)), cex.lab=1.2, cex.axis=1.2)
axis(2, at=seq(0, 60, 20), cex.axis=1.2)
axis(2, at=c(10, 30, 50), cex.axis=0.9)


#### PCA OF MD INDIV, FILTERED ####
MD_indiv_filtered <- MD_region_df[, regstowrite_md_indiv]
MD_indiv_filtered_pca <- PCA(MD_indiv_filtered, graph=FALSE, scale.unit=TRUE, ncp=ncol(MD_indiv_filtered))

plot(main="Scree plot: MD indiv, filtered", xlab="PC", ylab="Explained Variance %",
     x=c(1:nrow(MD_indiv_filtered_pca$eig)), y=MD_indiv_filtered_pca$eig[,2], col="red",
     pch=20, cex=2, xaxt="n", yaxt="n", cex.axis=1, cex.main=1.5)
axis(1, at=c(1, seq(5, length(MD_dual_filtered_pca$eig[,2]), 5)), cex.lab=1.2, cex.axis=1.2)
axis(2, at=c(0, 20, 40, 55), cex.axis=1.2)
axis(2, at=c(10, 30), cex.axis=0.9)

#### CUMULATIVE EXPLAINED VARIANCE PLOTS ####
par(mfrow=c(2,2))

plot(main="Cumulative Variance plot: FA dual, filtered", xlab="PC", ylab="Cumulative Variance %",
     x=c(1:ncol(FA_dual_filtered)), y=FA_dual_filtered_pca$eig[,3], col="blue",
     pch=20, cex=2, xaxt="n", yaxt="n", cex.axis=1, cex.main=1.5)
axis(1, at=seq(1, length(FA_dual_filtered_pca$eig[,3]), 1), cex.lab=1.2, cex.axis=1.2)
axis(2, at=c(seq(60, 100, 10), 100), cex.axis=0.9)

plot(main="Cumulative Variance plot: FA indiv., filtered", xlab="PC", ylab="Cumulative Variance %",
     x=c(1:ncol(FA_indiv_filtered)), y=FA_indiv_filtered_pca$eig[,3], col="blue",
     pch=20, cex=2, xaxt="n", cex.axis=1, cex.main=1.5)
axis(1, at=seq(1, length(FA_indiv_filtered_pca$eig[,3]), 1), cex.lab=1.2, cex.axis=1)
axis(2, at=c(seq(60, 100, 20), 100), cex.axis=0.9)

plot(main="Cumulative Variance plot: MD dual, filtered", xlab="PC", ylab="Cumulative Variance %",
     x=c(1:nrow(MD_dual_filtered_pca$eig)), y=MD_dual_filtered_pca$eig[,3], 
     col="red",pch=20, cex=2, xaxt="n", yaxt="n", cex.axis=1, cex.main=1.5)
axis(1, at=c(1, seq(5, length(MD_dual_filtered_pca$eig[,3]), 5)), cex.lab=1.2, cex.axis=1)
axis(2, at=seq(60, 100, 10), cex.lab=1, cex.axis=1)
axis(2, at=seq(70, 100, 20), cex.lab=1, cex.axis=1)

plot(main="Cumulative Variance plot: MD indiv, filtered", xlab="PC", ylab="Cumulative Variance %",
     x=c(1:nrow(MD_indiv_filtered_pca$eig)), y=MD_indiv_filtered_pca$eig[,3], col="red",
     pch=20, cex=2, xaxt="n", yaxt="n", cex.axis=1, cex.main=1.5)
axis(1, at=c(1, seq(5, length(MD_indiv_filtered_pca$eig[,3]), 5)), cex.lab=1.2, cex.axis=1.2)
axis(2, at=c(55, seq(60, 100, 20)), cex.lab=1, cex.axis=1.2)
axis(2, at=seq(70, 100, 20), cex.lab=1, cex.axis=1.2)

#### CHECK CONTRIBUTION OF EACH VARIABLE TO THE FIRST TO PCs ####

# FA individual, filtered
PC1_w_FA_indiv <- sort(FA_indiv_filtered_pca$var$contrib[,1], decreasing=FALSE)

# FA dual, filtered
PC1_w_FA_dual <- sort(FA_dual_filtered_pca$var$contrib[,1], decreasing=FALSE)

# MD individual,
PC1_w_MD_indiv <- sort(MD_indiv_filtered_pca$var$contrib[,1], decreasing=FALSE)

# MD dual, filtered
PC1_w_MD_dual <- sort(MD_dual_filtered_pca$var$contrib[,1], decreasing=FALSE)



#### PLOT CLASSES WITHIN PC1 AND PC2 ####
colizenak <- c("PC1", "PC2", "PC3")
palette <- c("red2","green2")
# FA dual
temp_df1 <- as.data.frame(FA_dual_filtered_pca$ind$coord)
colnames(temp_df1) <- colizenak
kolors <- palette[as.integer(1+FA_dual_region_df$Label)]
plot(temp_df1[,1:3], col=kolors, main="FA dual PC plots", cex=1.5, pch=20)
legend(0.047, 0.8, col=palette, legend = c("NC", "AD"), pch = c(rep(4,4),rep(20,3)), xpd=NA )

# FA indiv
temp_df2 <- as.data.frame(FA_indiv_filtered_pca$ind$coord)
colnames(temp_df2) <- colizenak
kolors <- palette[as.integer(1+FA_region_df$Label)]
plot(temp_df2[,1:3], col=kolors, main="FA individual PC plots", cex=1.5, pch=20)
legend(0.047, 0.8, col=palette, legend = c("NC", "AD"), pch = c(rep(4,4),rep(20,3)), xpd=NA )

# MA dual
temp_df3 <- as.data.frame(MD_dual_filtered_pca$ind$coord)
colnames(temp_df3) <- colizenak
kolors <- palette[as.integer(1+MD_dual_region_df$Label)]
plot(temp_df3[,1:3], col=kolors, main="MA dual PC plots", cex=1.5, pch=20)
legend(0.047, 0.83, col=palette, legend = c("NC", "AD"), pch = c(rep(4,4),rep(20,3)), xpd=NA )

# MA indiv
temp_df4 <- as.data.frame(MD_indiv_filtered_pca$ind$coord)
colnames(temp_df4) <- colizenak
kolors <- palette[as.integer(1+MD_region_df$Label)]
plot(temp_df4[,1:3], col=kolors, main="MA individual PC plots", cex=1.5, pch=20)
legend(0.047, 0.83, col=palette, legend = c("NC", "AD"), pch = c(rep(4,4),rep(20,3)), xpd=NA )


