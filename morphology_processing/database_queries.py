import os
import numpy as np
import allensdk.internal.core.lims_utilities as lu
from functools import partial
# NOTE: The queries in this file are only relevant for members of the allen institute. Functionality of this repository
# that depends on these queries will be limited for those from outside the allen institute


def default_query_engine():
    """Get Postgres query engine with environmental variable parameters"""

    return partial(
        lu.query,
        host=os.getenv("LIMS_HOST"),
        port=5432,
        database=os.getenv("LIMS_DBNAME"),
        user=os.getenv("LIMS_USER"),
        password=os.getenv("LIMS_PASSWORD")
    )


def query_for_63x_image_series_id(specimen_id,query_engine=None):
    if query_engine is None:
        query_engine = default_query_engine()

    query = f"""
    select max(id) as image_series_id from image_series
    where specimen_id = {specimen_id}
    group by specimen_id"""
    results = query_engine(query)
    return results[0]['image_series_id']


def query_for_63x_soma_coords(specimen_id,query_engine=None):

    if query_engine is None:
        query_engine = default_query_engine()

    imser_id_63x = query_for_63x_image_series_id(specimen_id,query_engine)

    sql_query = """
    select distinct 
                cell.id as cell_id, 
                ims63.id as image_series_63, 
                layert.name as layer_type, 
                si.specimen_tissue_index as z_index, 
                poly.path as poly_path
    from specimens cell
    join image_series ims63 on ims63.specimen_id = cell.id
    join sub_images si on si.image_series_id = ims63.id
    join avg_graphic_objects layer on layer.sub_image_id = si.id
    join avg_group_labels layert on layert.id = layer.group_label_id
    join avg_graphic_objects poly on poly.parent_id = layer.id
    where ims63.id = {}
    """.format(imser_id_63x)

    results = query_engine(sql_query)
    soma_only = [d['poly_path'] for d in results if d['layer_type'] == 'Soma']
    all_soma_coords = ",".join(soma_only)
    xs = [int(x) for x in all_soma_coords.split(",")[::2] if x != ""]
    ys = [int(y) for y in all_soma_coords.split(",")[1::2] if y != ""]

    return np.array(xs), np.array(ys)


def query_for_z_resolution(specimen_id,query_engine=None):
    if query_engine is None:
        query_engine = default_query_engine()

    sql = """
    select ss.id, ss.name, shs.thickness from specimens ss
    join specimen_blocks sb on ss.id = sb.specimen_id
    join blocks bs on bs.id = sb.block_id
    join thicknesses shs on shs.id = bs.thickness_id 
    where ss.id = {}
    """.format(specimen_id)
    result = query_engine(sql)[0]['thickness']

    return result
