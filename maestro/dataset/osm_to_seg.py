"""OSM to segmentation module."""

import json
from pathlib import Path

import cv2
import numpy as np


def map_category_to_meta(category: str) -> int:
    """Encode categories."""
    mapping = {
        # Buildings and Structures
        "building": 1,
        "flagpole": 1,
        "lighthouse": 1,
        "obelisk": 1,
        "observatory": 1,
        # Transportation Infrastructure
        "aerialway_pylon": 2,
        "airport": 2,
        "gas_station": 2,
        "helipad": 2,
        "parking": 2,
        "road": 2,
        "runway": 2,
        "taxiway": 2,
        # Industrial and Energy Infrastructure (chimneys go here)
        "chimney": 3,
        "petroleum_well": 3,
        "power_plant": 3,
        "power_substation": 3,
        "power_tower": 3,
        "satellite_dish": 3,
        "silo": 3,
        "storage_tank": 3,
        "wind_turbine": 3,
        "works": 3,
        # Water Features
        "river": 4,
        "fountain": 4,
        # Other
        "leisure": 5,
    }
    return mapping.get(category, 5)


def process_geojson_file(path: str | Path) -> np.ndarray:  # noqa: C901
    """Create raster from geojson file."""
    image = np.zeros((512, 512), dtype=np.uint8)

    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
        if not data:
            return image
        for feature in data.get("features", []):
            properties = feature.get("properties", {})
            category = properties.get("category")

            # Map category to meta category value
            meta_value = map_category_to_meta(category)

            geometry = feature.get("geometry", {})
            geom_type = geometry.get("type")
            coordinates = geometry.get("coordinates", [])

            match geom_type:
                case "LineString":
                    # Draw lines on the image
                    points = np.array(coordinates, dtype=np.int32)
                    # Ensure points are within image bounds
                    points = np.clip(points, 0, 511)
                    # Draw the line with thickness 1
                    cv2.polylines(
                        image,
                        [points],
                        isClosed=False,
                        color=meta_value,
                        thickness=2,
                    )

                case "MultiLineString":
                    # Draw multiple lines
                    for line in coordinates:
                        points = np.array(line, dtype=np.int32)
                        points = np.clip(points, 0, 511)
                        cv2.polylines(
                            image,
                            [points],
                            isClosed=False,
                            color=meta_value,
                            thickness=2,
                        )

                case "Polygon":
                    # Draw filled polygon on the image
                    # Polygons may have multiple rings ; \
                    # coordinates[0] is the exterior ring
                    for polygon in coordinates:
                        points = np.array(polygon, dtype=np.int32)
                        # Ensure points are within image bounds
                        points = np.clip(points, 0, 511)
                        if np.size(points) != 0:
                            cv2.fillPoly(image, [points], color=meta_value)

                case "MultiPolygon":
                    # Draw multiple polygons
                    for multipolygon in coordinates:
                        for polygon in multipolygon:
                            points = np.array(polygon, dtype=np.int32)
                            points = np.clip(points, 0, 511)
                            if np.size(points) != 0:
                                cv2.fillPoly(image, [points], color=meta_value)

                case "Point":
                    # Draw a point on the image
                    point = np.array(coordinates, dtype=np.int32)
                    point = np.clip(point, 0, 511)
                    # Draw a small circle to represent the point
                    cv2.circle(
                        image,
                        tuple(point),
                        radius=1,
                        color=meta_value,
                        thickness=-1,
                    )

                case "MultiPoint":
                    # Draw multiple points
                    for coord in coordinates:
                        point = np.array(coord, dtype=np.int32)
                        point = np.clip(point, 0, 511)
                        cv2.circle(
                            image,
                            tuple(point),
                            radius=1,
                            color=meta_value,
                            thickness=-1,
                        )

                case _:
                    msg = f"Unsupported geometry type '{geom_type}' in file {path}"
                    raise ValueError(msg)

    return image
