# Assessment tool

This package recommends coordinate precision from local population density and a target anonymity-set size. It reads a Point GeoJSON file and a population-count GeoTIFF, then writes an assessed Point GeoJSON file.

```bash
python -m assessment \
  --input ../CoordinateJSONs/Synthetic/US.geojson \
  --raster path/to/worldpop_population.tif \
  --output assessed_locations.geojson \
  --k 40
```

Download a suitable population-count GeoTIFF from the official [WorldPop data catalog](https://www.worldpop.org/datacatalog/) or its [population-count listings](https://hub.worldpop.org/geodata/listing?id=29). Choose a product, year, and resolution covering the input points.

For the complete option reference and the other workflows, see the repository [README](../README.md).
