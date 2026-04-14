##  02_prune_tree.R
##
##  Prune the full species-level vascular plant tree down to one tip per genus.
##  Tip labels have the form  Order_Family_Genus_species.
##  For each genus the retained tip is chosen by minimum terminal branch length
##  (most "average" representative); ties broken by first occurrence.
##
##  Input:  data/best_wcvp.tre_dated
##  Output: data/genus_tree.tre

library(ape)

## ---- load ------------------------------------------------------------------
cat("Reading tree …\n")
phy <- read.tree("data/best_wcvp.tre_dated")
cat(sprintf("  %d tips, %d internal nodes\n", Nnode(phy) + 1L,
            Nnode(phy)))

## ---- parse genus from tip labels -------------------------------------------
## Format: Order_Family_Genus_species  (underscore-delimited, field 3)
tip_parts <- strsplit(phy$tip.label, "_", fixed = TRUE)
genera     <- vapply(tip_parts, function(x) x[3L], character(1L))

## Sanity check: flag any tips that don't match the expected 4-field pattern
bad <- which(lengths(tip_parts) != 4L)
if (length(bad)) {
  warning(sprintf(
    "%d tip(s) did not have exactly 4 underscore-delimited fields; genus set to NA.\n  e.g. %s",
    length(bad), phy$tip.label[bad[1]]
  ))
}

## ---- crown age per genus from the full tree --------------------------------
nh       <- node.depth.edgelength(phy)
max_nh   <- max(nh[seq_along(phy$tip.label)])
node_age <- max_nh - nh

genus_list <- split(which(!is.na(genera)), genera[!is.na(genera)])
crown_ages <- vapply(genus_list, function(idx) {
  if (length(idx) == 1L) 0.0
  else node_age[getMRCA(phy, idx)]
}, numeric(1L))

crown_out <- "data/genus_crown_ages.csv"
write.csv(data.frame(genus = names(crown_ages), crown_age = crown_ages),
          crown_out, row.names = FALSE)
cat(sprintf("  Crown ages saved → %s\n", crown_out))

## ---- pick one representative tip per genus ---------------------------------
## Use terminal (pendant) branch length as the selection criterion:
## the tip with the shortest pendant branch is least likely to be a
## long-branch outlier.  Among equal lengths, take the first tip.
pendant_bl <- phy$edge.length[match(seq_along(phy$tip.label),
                                    phy$edge[, 2L])]

tip_df <- data.frame(
  label  = phy$tip.label,
  genus  = genera,
  bl     = pendant_bl,
  stringsAsFactors = FALSE
)
tip_df <- tip_df[!is.na(tip_df$genus), ]           # drop any NA-genus tips

## Within each genus keep the tip with the smallest pendant branch length
tip_df <- tip_df[order(tip_df$bl), ]               # sort ascending
keepers <- tip_df[!duplicated(tip_df$genus), ]     # first (= shortest) per genus

cat(sprintf("  %d unique genera found → retaining %d tips\n",
            nrow(keepers), nrow(keepers)))

## ---- prune -----------------------------------------------------------------
drop_tips <- setdiff(phy$tip.label, keepers$label)
cat(sprintf("  Dropping %d tips …\n", length(drop_tips)))
pruned <- drop.tip(phy, drop_tips)

## ---- rename tips to genus name only ----------------------------------------
old_to_genus <- setNames(keepers$genus, keepers$label)
pruned$tip.label <- old_to_genus[pruned$tip.label]

cat(sprintf("  Pruned tree: %d tips, %d internal nodes\n",
            length(pruned$tip.label), Nnode(pruned)))

## ---- write -----------------------------------------------------------------
out <- "data/genus_tree.tre"
write.tree(pruned, file = out)
cat(sprintf("Saved → %s\n", out))