library(ggplot2)
library(dplyr)

d <- read.csv(file="../model/mushrooms_v2.csv", header=TRUE, sep=",")
ggplot(data.frame(animals), aes(x=animals)) +
  geom_bar()

d %>%
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~key, scales = "free") +
  geom_bar(position="dodge", stat="identity")
