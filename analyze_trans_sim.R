library(ggplot2)
library(reshape)
library(grid)
library(dplyr)

############################################################


f_sim <- "out_sim_only.csv"
data_sim <- read.csv(f_sim)


f_trans <- "out_prior2.csv"
data_trans <- read.csv(f_trans)


data_trans$ID <- seq.int(nrow(data_trans))
data_sim$ID <- seq.int(nrow(data_sim))

colnames(data_sim)[which(names(data_sim) == "training")] <- "from"
colnames(data_sim)[which(names(data_sim) == "transfer")] <- "to"

############################################################

to_binary <- function(x) {
	x <- as.character(x)
	n = 0
	for (i in 1:nchar(x)) {
		if (substr(x,i,i) == "b") {
			n = n + 2**i
		}

	}
	return(n)
}


d.trans.1 <- data_trans %>%
			 group_by(from,to) %>%
			 top_n(n=1, wt=prior) %>% 
			 top_n(n=1,wt=ID) %>%
			 ungroup %>%
		   	#mutate(factor_ord = to_binary(from) + to_binary(to)) %>%

		  	#transform(from=factor(reorder(from, factor_ord))) %>%
			 select(from,to,prior) 




d.sim.1 <- data_sim %>%	

			group_by(from,to) %>%

			mutate(mean_sim=mean(similarity)) %>%
			mutate(sd_sim=sd(similarity)) %>%
			top_n(n=1, wt=ID) %>%

		  # ungroup %>%
		  # mutate(factor_ord = to_binary(from)) %>%
		  #transform(from=factor(reorder(from, factor_ord))) %>%
		   select(from,to,mean_sim) 


#head(d.sim.1)
#head(d.sim.1)

#d.trans.1$from <- as.character(d.trans.1$from)
#d.trans.1$to <- as.character(d.trans.1$to)
#d.sim.1$from <- as.character(d.sim.1$from)
#d.sim.1$to <- as.character(d.sim.1$to)

froms <- levels(d.sim.1$from)
tos <- levels(d.sim.1$to)

x=0
for (i in 1:length(froms)) {
	for (j in 1:length(tos)) {
		x = x + 1
		f <- froms[i]
		t <- tos[i]
		print(paste(f,t))

		prior_ind <- which(grepl(f,d.sim.1$from) &grepl(t,d.sim.1$to))

		print(prior_ind)
	}
}

#print(x)