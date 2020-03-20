# Load igraph library (assuming installed, else do install.package("igraph"))
message("Load igraph library...")
library(igraph)

loadInfomap <- function() {
  message("Loading Infomap...")
  dyn.load(paste("infomap/infomap", .Platform$dynlib.ext, sep=""))
  source("infomap/infomap.R")
  # The cacheMetaData(1) will cause R to refresh its object tables. Without it, inheritance of wrapped objects may fail.
  cacheMetaData(1)
}


# Only load if not already loaded
if (!exists("HierarchicalNetwork")) {
  loadInfomap()
} else {
  message("Infomap already loaded!")
}


# Load Infomap library
source("load-infomap")

## Zachary's karate club
g <- graph.famous("Zachary")

# Init Infomap network
infomap <- Infomap("--two-level --silent")

# Add links to Infomap network from igraph data
edgelist <- get.edgelist(g)
apply(edgelist, 1, function(e) infomap$addLink(e[1] - 1, e[2] - 1))

infomap$run()

tree <- infomap$tree

cat("Partitioned network in", tree$numTopModules(), "modules with codelength", tree$codelength(), "bits:\n")

clusterIndexLevel <- 1 # 1, 2, ... or -1 for top, second, ... or lowest cluster level
leafIt <- tree$leafIter(clusterIndexLevel)
modules <- integer(length = network$numNodes())

while (!leafIt$isEnd()) {
  modules[leafIt$originalLeafIndex + 1] = leafIt$moduleIndex() + 1
  leafIt$stepForward()
}

# Create igraph community data
comm <- make_clusters(g, membership = modules, algorithm = 'Infomap')
print(comm)

# Plot communities and network
plot(comm, g)
