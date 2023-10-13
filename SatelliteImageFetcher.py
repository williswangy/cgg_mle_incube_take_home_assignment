import logging
import pystac_client
import planetary_computer
import rasterio
from rasterio import windows, features, warp
from PIL import Image
from pystac.extensions.eo import EOExtension as eo
import json
import numpy as np
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('config.json', 'r') as file:
    config = json.load(file)

STAC_URL = config['STAC_URL']
COLLECTION_NAME = config['COLLECTION_NAME']
TIME_OF_INTEREST = config['TIME_OF_INTEREST']
TARGET_WIDTH = config['TARGET_WIDTH']



def fetch_least_cloudy_image(aoi, time_of_interest=TIME_OF_INTEREST):
    """
    Fetches the asset href and details of the least cloudy image for a specified area of interest (AOI) within a given time frame.

    Args:
        aoi (dict): The area of interest, typically a Polygon in GeoJSON format.
        time_of_interest (str): Time frame in ISO format. Defaults to `TIME_OF_INTEREST` from config.

    Returns:
        tuple: A tuple containing the href link to the asset and the dictionary with item details.
    """

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    search = catalog.search(
        collections=[COLLECTION_NAME],
        intersects=aoi,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 10}},
    )

    items = search.item_collection()
    logger.info(f"Returned {len(items)} Items")

    least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)
    logger.info(
        f"Choosing {least_cloudy_item.id} from {least_cloudy_item.datetime.date()}"
        f" with {eo.ext(least_cloudy_item).cloud_cover}% cloud cover"
    )

    return least_cloudy_item.assets["visual"].href, least_cloudy_item.to_dict()


def read_image_from_asset(asset_href, aoi):
    """
    Reads and returns a resized image from a given asset href for the specified area of interest (AOI).

    Args:
        asset_href (str): The href link to the asset.
        aoi (dict): The area of interest, typically a Polygon in GeoJSON format.

    Returns:
        PIL.Image: The resized image of the AOI from the asset.
    """

    logger.info(f"Opening asset: {asset_href}")

    with rasterio.open(asset_href) as ds:
        logger.info(f"Asset CRS: {ds.crs}")

        aoi_bounds = features.bounds(aoi)
        logger.info(f"Original AOI bounds: {aoi_bounds}")

        warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
        logger.info(f"Warped AOI bounds: {warped_aoi_bounds}")

        aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
        logger.info(f"AOI window: {aoi_window}")

        band_data = ds.read(window=aoi_window)
        logger.info(f"Band data shape: {band_data.shape}")

    img = Image.fromarray(np.transpose(band_data, axes=[1, 2, 0]))
    w, h = img.size
    aspect = w / h
    target_w = 800
    target_h = int(target_w / aspect)
    logger.info(f"Resizing image to width {target_w} and height {target_h}")

    return img.resize((target_w, target_h), Image.Resampling.BILINEAR)


def save_image(img, item_details, save_folder='saved_images'):
    """
    Saves the provided image in the specified folder with a name based on item details.

    Args:
        img (PIL.Image): Image to be saved.
        item_details (dict): Dictionary with details of the satellite image.
        save_folder (str): Folder to save the image in.

    Returns:
        str: Path where the image was saved.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Generate an image name based on item ID and date
    img_name = f"{item_details['id']}.png"
    img_path = os.path.join(save_folder, img_name)

    img.save(img_path, 'PNG')

    logger.info(f"Image saved at: {img_path}")
    return img_path


if __name__ == "__main__":
    AREA_OF_INTEREST = {
        "type": "Polygon",
        "coordinates": [
            [
                [-148.56536865234375, 60.80072385643073],
                [-147.44338989257812, 60.80072385643073],
                [-147.44338989257812, 61.18363894915102],
                [-148.56536865234375, 61.18363894915102],
                [-148.56536865234375, 60.80072385643073],
            ]
        ],
    }

    asset_href, item_details = fetch_least_cloudy_image(AREA_OF_INTEREST)
    img = read_image_from_asset(asset_href, AREA_OF_INTEREST)
    save_image(img, item_details)
