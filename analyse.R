 require(lattice)
 require(ggplot2)
 traj <- read.csv("trajectories.csv", header=TRUE)
 landscape <- read.csv("landscape.csv", header=TRUE)
 # Print trajectories
 png("trajectories.png")
 print(with(traj, xyplot(X0 ~ X1, groups=Agent, type='l', auto.key=TRUE)))
 dev.off()
 # True fitness Landscape
 png("landscape.png")
 print(wireframe(utility ~ X0*X1, landscape, zlim=c(0, 2), colorkey=TRUE, drape=TRUE))
 dev.off()
 # Starting difference
 landscape <- read.csv("start_l.csv", header=TRUE)
 png("start_landscape.png")
 print(wireframe(utility ~ X0*X1, landscape, zlim=c(0, 1), colorkey=TRUE, drape=TRUE))
 dev.off()
 # Ending difference
 landscape <- read.csv("end_l.csv", header=TRUE)
 png("end_landscape.png")
 print(wireframe(utility ~ X0*X1, landscape, zlim=c(0, 1), colorkey=TRUE, drape=TRUE))
 dev.off()

 # Replicated q consensus
q_conv <- read.csv("q_convergence_rep.csv", header=TRUE) 
png("q_convergence_rep.png")
c <- ggplot(q_conv, aes(q, time))
print(c + stat_smooth())
dev.off()

 # Replicated indiv consensus
indiv_conv <- read.csv("individual_convergence_rep.csv", header=TRUE) 
png("individual_convergence_rep.png")
c <- ggplot(indiv_conv, aes(q, convergence))
print(c + stat_smooth())
dev.off()

# Pair discussion individual convergence
pair_indiv_conv <- read.csv("individual_pair_convergence_rep.csv", header=TRUE) 
png("individual_pair_convergence_rep.png")
c <- ggplot(pair_indiv_conv, aes(q, convergence))
print(c + stat_smooth())
dev.off()

# Headless chicken individual convergence
chkn_indiv_conv <- read.csv("individual_chicken_convergence_rep.csv", header=TRUE) 
png("individual_chicken_convergence_rep.png")
c <- ggplot(chkn_indiv_conv, aes(q, convergence))
print(c + stat_smooth())
dev.off()

# Pair discussion individual convergence
pair_indiv_conv <- read.csv("individual_pair_convergence_rep.csv", header=TRUE) 
png("individual_pair_convergence_rep.png")
c <- ggplot(pair_indiv_conv, aes(q, convergence))
print(c + stat_smooth())
dev.off()

# Pair random discussion individual convergence
pair_rand_conv <- read.csv("pair_random_convergence.csv", header=TRUE) 
png("pair_random_convergence.png")
c <- ggplot(pair_rand_conv, aes(q, convergence))
print(c + stat_smooth())
dev.off()

# Control discussion individual convergence
control_conv <- read.csv("control_individual_convergence.csv", header=TRUE) 
png("control_individual_convergence.png")
c <- ggplot(control_conv, aes(q, convergence))
print(c + stat_smooth())
dev.off()

#Chicken time to consensus
struct_pair_convergence <- read.csv("struct_pair_convergence.csv", header=TRUE) 
png("struct_pair_convergence.png")
c <- ggplot(struct_pair_convergence, aes(q, convergence))
print(c + stat_smooth())
dev.off()

#Join up the individual convergence data sets
pair_indiv_conv$protocol = "Pairwise"
chkn_indiv_conv$protocol = "Headless Chicken"
struct_pair_convergence$protocol = "Structured discussion"
indiv_conv$protocol = "Standard"
results = merge(pair_indiv_conv, chkn_indiv_conv, all=TRUE)
results = merge(results, struct_pair_convergence, all=TRUE)
results = merge(results, indiv_conv, all=TRUE)
results = merge(results, pair_rand_conv, all=TRUE)
results = merge(results, control_conv, all=TRUE)

#Plot on same graph
png("pair_convergence.png")
c <- ggplot(results, aes(x=q, y=convergence, color=protocol))
print(c + stat_smooth())
dev.off()

# Get aggregates
agg_res = with(results, aggregate(results, by=list(q,protocol), FUN=mean))
png("pair_convergence_means.png")
print(with(agg_res, xyplot(convergence ~ q, group=Group.2, type='l', auto.key=TRUE)))
dev.off()

# Get within discussion records
within_res <- read.csv("within_discussion_convergence.csv", header=TRUE)
png("within_convergence.png")
c <- ggplot(within_res, aes(x=step, y=convergence, color=protocol))
print(c + stat_smooth())
dev.off()

# Plot means
agg_within = with(within_res, aggregate(within_res, by=list(step, protocol), FUN=mean))
png("within_convergence_means.png")
print(with(agg_within, xyplot(convergence ~ step, group=Group.2, type='l', auto.key=TRUE)))
dev.off()

# Utils

utils <- read.csv("true_util.csv", header=TRUE)
png("true_util.png")
c <- ggplot(utils, aes(x=q, y=utility, color=protocol))
print(c + stat_smooth())
dev.off()

# Get aggregates
agg_utils = with(utils, aggregate(utils, by=list(q,protocol), FUN=mean))
png("true_util_means.png")
print(with(agg_utils, xyplot(utility ~ q, group=Group.2, type='l', auto.key=TRUE)))
dev.off()

# Fidelity to true landscape
fidelity <- read.csv("fidelity.csv", header=TRUE)
png("fidelity.png")
c <- ggplot(fidelity, aes(x=q, y=fidelity, color=protocol))
print(c + stat_smooth())
dev.off()