install.packages("BradleyTerry2")

# Get paired comparison data
data = read.csv('popfss_comparison_results.csv')
summary(data)
names(data)

# Set data to correct formats for model input
outcome = matrix(c(data$left_wins, data$right_wins), nrow=length(data$left_wins), ncol=2, byrow=FALSE)
player1 = data$left_subject
player2 = data$right_subject

# Fit Bradley Terry model
fit = BradleyTerry2::BTm(outcome, player1, player2)

# Save model fit to csv
write.csv(fit$coefficients, file="popfss_model_fit_r.csv")

############
# Repeat for STA and STB separately

data = read.csv('popfss_comparison_results_sta.csv')
outcome = matrix(c(data$left_wins, data$right_wins), nrow=length(data$left_wins), ncol=2, byrow=FALSE)
player1 = data$left_subject
player2 = data$right_subject
fit = BradleyTerry2::BTm(outcome, player1, player2)
write.csv(fit$coefficients, file="popfss_model_fit_r_sta.csv")

data = read.csv('popfss_comparison_results_stb.csv')
outcome = matrix(c(data$left_wins, data$right_wins), nrow=length(data$left_wins), ncol=2, byrow=FALSE)
player1 = data$left_subject
player2 = data$right_subject
fit = BradleyTerry2::BTm(outcome, player1, player2)
write.csv(fit$coefficients, file="popfss_model_fit_r_stb.csv")

############
# Repeat, setting reference image

# Image default
ref = "ssw_067_helcats_HCME_B__20131128_02_stb_diff_20131129_005001.jpg"
outcome = matrix(c(data$left_wins, data$right_wins), nrow=length(data$left_wins), ncol=2, byrow=FALSE)
player1 = data$left_subject
player2 = data$right_subject
fit = BradleyTerry2::BTm(outcome, player1, player2, refcat=ref)
write.csv(fit$coefficients, file="popfss_model_fit_r_ssw_067.csv")

# Pick 3 different images
ref = "ssw_204_helcats_HCME_B__20120831_01_stb_diff_20120901_052941.jpg"
outcome = matrix(c(data$left_wins, data$right_wins), nrow=length(data$left_wins), ncol=2, byrow=FALSE)
player1 = data$left_subject
player2 = data$right_subject
fit = BradleyTerry2::BTm(outcome, player1, player2, refcat=ref)
write.csv(fit$coefficients, file="popfss_model_fit_r_ssw_204.csv")

ref = "ssw_1128_helcats_HCME_A__20091028_01_sta_diff_20091029_084901.jpg"
outcome = matrix(c(data$left_wins, data$right_wins), nrow=length(data$left_wins), ncol=2, byrow=FALSE)
player1 = data$left_subject
player2 = data$right_subject
fit = BradleyTerry2::BTm(outcome, player1, player2, refcat=ref)
write.csv(fit$coefficients, file="popfss_model_fit_r_ssw_1128.csv")

ref = "ssw_1193_helcats_HCME_A__20080602_01_sta_diff_20080603_084901.jpg"
outcome = matrix(c(data$left_wins, data$right_wins), nrow=length(data$left_wins), ncol=2, byrow=FALSE)
player1 = data$left_subject
player2 = data$right_subject
fit = BradleyTerry2::BTm(outcome, player1, player2, refcat=ref)
write.csv(fit$coefficients, file="popfss_model_fit_r_ssw_1193.csv")



