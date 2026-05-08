# Load required libraries
library(ggplot2)
library(dplyr)
library(readr)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(rnaturalearthhires)  # for higher-res borders (install if needed)
library(viridis)
library(ggspatial)           # for scale bar & north arrow

# Read your .tsv file
spiders <- read_tsv("D:/02_School/MSc/02_Sem2627-2/MultiModal/MultiModal_Assignment/input/CrypticBio-Invasive_continent.tsv")

# Count unique species
unique_species <- unique(spiders$scientificName)
n_species <- length(unique_species)
cat("Number of species:", n_species, "\n")
n_records <- nrow(spiders)
cat("Number of records:", n_records, "\n")

# Get world map layers
world      <- ne_countries(scale = "medium", returnclass = "sf")
graticules <- st_graticule(ndiscr = 10000, margin = 0.01)

# ── Build plot ──────────────────────────────────────────────────────────────
p <- ggplot() +
  
  # Subtle ocean fill
  geom_sf(data = graticules,
          color = "#d0dce8", linewidth = 0.15, alpha = 0.6) +
  
  # Land fill — clean off-white
  geom_sf(data = world,
          fill  = "#f0ede8",
          color = "#9aabb8",   # country border color
          linewidth = 0.25) +
  
  # Occurrence points, colored by species
  geom_point(data = spiders,
             aes(x = decimalLongitude,
                 y = decimalLatitude,
                 color = scientificName),
             alpha = 0.75,
             size  = 1.4,
             shape = 16,
             show.legend = FALSE) +
  
  # Viridis turbo palette (perceptually distinct across many species)
  scale_color_viridis_d(option = "turbo") +
  
  # Robinson projection — standard for global biodiversity maps
  coord_sf(crs    = "+proj=robin",
           xlim   = c(-180, 180),
           ylim   = c(-55, 90),
           expand = FALSE,
           default_crs = sf::st_crs(4326)) + # ← tells ggplot xlim/ylim are in degrees
  
  # Annotation: species & record counts (bottom-left)
  annotate("text",
           x = -15500000, y = -5400000,
           label = paste0(n_species, " cryptic spider species  ·  ",
                          format(n_records, big.mark = ","), " occurrence records"),
           hjust = 0, vjust = 0,
           size  = 3.2,
           color = "#3a4a5a",
           family = "sans") +
  
  # Labels
  labs(
    title    = "",
    subtitle = "",
    caption  = "Invasive Species"
  ) +
  
  theme_void(base_family = "sans") +
  theme(
    # White / near-white backgrounds
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "#dce8f0", color = NA),  # ocean blue-grey
    
    # Title block
    plot.title    = element_text(size = 15, face = "bold",
                                 color = "#1a2a3a", margin = margin(b = 4)),
    plot.subtitle = element_text(size =  9, color = "#4a5a6a",
                                 margin = margin(b = 6)),
    plot.caption  = element_text(size =  7.5, color = "#7a8a9a",
                                 margin = margin(t = 6)),
    
    # Generous margins
    plot.margin = margin(14, 14, 10, 14)
  )

p 
# ── Save ────────────────────────────────────────────────────────────────────
ggsave("Invasive.png",
       plot   = p,
       width  = 16,
       height = 9,
       dpi    = 300,
       bg     = "white")


