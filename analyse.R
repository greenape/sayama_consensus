 require(lattice)
 require(ggplot2)
 sink("analysis.log")
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

#'q', 'run', 'utility','max_util','fidelity','convergence', 'protocol','dimensions'
twod <- read.csv("2d_mp.csv", header=TRUE)
#twod <- with(twod, subset(twod, protocol == 'Standard' | protocol == 'Structured'))
#supl <- read.csv("2d_fix.csv", header=TRUE)
#twod = merge(twod, supl, all=TRUE)
twod$max_sum = 1

png("fidelity_2d.png")
c <- ggplot(twod, aes(x=q, y=fidelity, color=protocol))
print(c + stat_smooth())
dev.off()

png("convergence_2d.png")
c <- ggplot(twod, aes(x=q, y=convergence, color=protocol))
print(c + stat_smooth())
dev.off()

png("util_2d.png")
c <- ggplot(twod, aes(x=q, y=utility, color=protocol))
print(c + stat_smooth())
dev.off()

#'q', 'run', 'utility','max_util','fidelity','convergence', 'protocol','dimensions'
twod_6p <- read.csv("2d_mp_6_p.csv", header=TRUE)
#twod <- with(twod, subset(twod, protocol == 'Standard' | protocol == 'Structured'))
#supl <- read.csv("2d_fix.csv", header=TRUE)
#twod = merge(twod, supl, all=TRUE)
twod_6p$max_sum = 1

png("fidelity_2d_6p.png")
c <- ggplot(twod_6p, aes(x=q, y=fidelity, color=protocol))
print(c + stat_smooth())
dev.off()

png("convergence_2d_6p.png")
c <- ggplot(twod_6p, aes(x=q, y=convergence, color=protocol))
print(c + stat_smooth())
dev.off()

png("util_2d_6p.png")
c <- ggplot(twod_6p, aes(x=q, y=utility, color=protocol))
print(c + stat_smooth())
dev.off()

# Boxplots..

png("fidelity_2d_box_6p.png")
with(twod_6p, boxplot(fidelity ~ q, group=protocol, auto.key=TRUE))
dev.off()

#'q', 'run', 'utility','max_util','fidelity','convergence', 'protocol','dimensions'
twod_6p_long <- read.csv("2d_mp_6_p_long.csv", header=TRUE)
#twod <- with(twod, subset(twod, protocol == 'Standard' | protocol == 'Structured'))
#supl <- read.csv("2d_fix.csv", header=TRUE)
#twod = merge(twod, supl, all=TRUE)

png("fidelity_2d_6p_long.png")
c <- ggplot(twod_6p_long, aes(x=q, y=fidelity, color=protocol))
print(c + stat_smooth())
dev.off()

png("convergence_2d_6p_long.png")
c <- ggplot(twod_6p_long, aes(x=q, y=convergence, color=protocol))
print(c + stat_smooth())
dev.off()

png("util_2d_6p_long.png")
c <- ggplot(twod_6p_long, aes(x=q, y=utility, color=protocol))
print(c + stat_smooth())
dev.off()


convergence_fixed <- read.csv("convergence_fixed.csv", header=TRUE)
#twod <- with(twod, subset(twod, protocol == 'Standard' | protocol == 'Structured'))
#supl <- read.csv("2d_fix.csv", header=TRUE)
#twod = merge(twod, supl, all=TRUE)

png("fidelity_convergence_fixed.png")
c <- ggplot(convergence_fixed, aes(x=q, y=fidelity, color=protocol))
print(c + stat_smooth())
dev.off()

png("convergence_convergence_fixed.png")
c <- ggplot(convergence_fixed, aes(x=q, y=convergence, color=protocol))
print(c + stat_smooth())
dev.off()

png("util_convergence_fixed.png")
c <- ggplot(convergence_fixed, aes(x=q, y=utility, color=protocol))
print(c + stat_smooth())
dev.off()

convergence_fixed_slow <- read.csv("convergence_fixed_slow.csv", header=TRUE)
#twod <- with(twod, subset(twod, protocol == 'Standard' | protocol == 'Structured'))
#supl <- read.csv("2d_fix.csv", header=TRUE)
#twod = merge(twod, supl, all=TRUE)

png("fidelity_convergence_fixed_slow.png")
c <- ggplot(convergence_fixed_slow, aes(x=q, y=fidelity, color=protocol))
print(c + stat_smooth())
dev.off()

png("convergence_convergence_fixed_slow.png")
c <- ggplot(convergence_fixed_slow, aes(x=q, y=convergence, color=protocol))
print(c + stat_smooth())
dev.off()

png("util_convergence_fixed_slow.png")
c <- ggplot(convergence_fixed_slow, aes(x=q, y=utility, color=protocol))
print(c + stat_smooth())
dev.off()

convergence_fixed_paired <- read.csv("convergence_fixed_paired.csv", header=TRUE)
#twod <- with(twod, subset(twod, protocol == 'Standard' | protocol == 'Structured'))
#supl <- read.csv("2d_fix.csv", header=TRUE)
#twod = merge(twod, supl, all=TRUE)

png("fidelity_convergence_fixed_paired.png")
c <- ggplot(convergence_fixed_paired, aes(x=q, y=fidelity, color=protocol))
print(c + stat_smooth())
dev.off()

png("convergence_convergence_fixed_paired.png")
c <- ggplot(convergence_fixed_paired, aes(x=q, y=convergence, color=protocol))
print(c + stat_smooth())
dev.off()

png("util_convergence_fixed_paired.png")
c <- ggplot(convergence_fixed_paired, aes(x=q, y=utility, color=protocol))
print(c + stat_smooth())
dev.off()

# Now some stats

# Summary stats on everything

print("2d, 3player, 100 iterations")
print("utility & fidelity")
print(with(twod, cor(utility, fidelity)))
print("utility & convergence")
print(with(twod, cor(utility, convergence)))
print("utility & q")
print(with(twod, cor(utility, q)))
print("fidelity & convergence")
print(with(twod, cor(fidelity, convergence)))
struct = with(twod, subset(twod, protocol == 'Structured'))
standard = with(twod, subset(twod, protocol == 'Standard'))
ctrl = with(twod, subset(twod, protocol == 'Control'))
print("fidelity & convergence structured")
print(with(struct, cor(fidelity, convergence)))
print("fidelity & convergence standard")
print(with(standard, cor(fidelity, convergence)))
print("fidelity & convergence control")
print(with(ctrl, cor(fidelity, convergence)))

print("AOV q & protocol")
a = with(twod, aov(utility ~ q + protocol + q*protocol))
print(summary(a))

l = with(twod, lm(utility ~ q + protocol + q*protocol + q*fidelity + q*convergence))
print("lm")
print(summary(l))

l = with(twod, lm(fidelity ~ q + protocol + q*protocol + q*convergence))
print("lm")
print(summary(l))

print("2d, 6player, 100 iterations")
print("utility & fidelity")
print(with(twod_6p, cor(utility, fidelity)))
print("utility & convergence")
print(with(twod_6p, cor(utility, convergence)))
print("utility & q")
print(with(twod_6p, cor(utility, q)))
print("fidelity & convergence")
print(with(twod_6p, cor(fidelity, convergence)))
struct = with(twod_6p, subset(twod_6p, protocol == 'Structured'))
standard = with(twod_6p, subset(twod_6p, protocol == 'Standard'))
ctrl = with(twod_6p, subset(twod_6p, protocol == 'Control'))
print("fidelity & convergence structured")
print(with(struct, cor(fidelity, convergence)))
print("fidelity & convergence standard")
print(with(standard, cor(fidelity, convergence)))
print("fidelity & convergence control")
print(with(ctrl, cor(fidelity, convergence)))

print("AOV q & protocol")
a = with(twod_6p, aov(utility ~ q + protocol + q*protocol))
print(summary(a))

l = with(twod_6p, lm(utility ~ q + protocol + q*protocol + q*fidelity + q*convergence, data=twod_6p))
print("lm")
print(summary(l))

l = with(twod_6p, lm(fidelity ~ q + protocol + q*protocol + q*convergence))
print("lm")
print(summary(l))

# ANOVAs


# Regression

# Hypothesis tests

conv_struct = with(convergence_fixed, subset(convergence_fixed, protocol == 'Structured'))
conv_st = with(convergence_fixed, subset(convergence_fixed, protocol == 'Standard'))
conv_ctrl = with(convergence_fixed, subset(convergence_fixed, protocol == 'Control'))
print(t.test(conv_struct$convergence, conv_st$convergence, alt="less"))
print(t.test(conv_ctrl$convergence, conv_st$convergence, alt="less"))

# Linear models
print("Fidelity")
l = with(convergence_fixed, lm(fidelity ~ q + protocol + q*protocol))
a = with(convergence_fixed, aov(fidelity ~ q + protocol + q*protocol + Error(run)))
lmin = step(l)
print(summary(l))
print(summary(lmin))
print(summary(a))

print("Utility")
l = with(convergence_fixed, lm(utility ~ q + protocol + q*protocol))
a = with(convergence_fixed, aov(utility ~ q + protocol + q*protocol + Error(run)))
lmin = step(l)
print(summary(l))
print(summary(lmin))
print(summary(a))

print("Convergence")
l = with(convergence_fixed, lm(convergence ~ q + protocol + q*protocol))
a = with(convergence_fixed, aov(convergence ~ q + protocol + q*protocol + Error(run)))
lmin = step(l)
print(summary(l))
print(summary(lmin))
print(summary(a))

sink(NULL)