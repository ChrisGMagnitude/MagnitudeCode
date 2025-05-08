import numpy as np
import math
import os
import matplotlib.pyplot as plt
import psycopg2, osgeo.ogr, osgeo.gdal, osgeo.osr
import geopandas as gp

def get_class_definitions():
    background = {}
    background['Description'] = 'background'
    background['Include'] = [7,8,9]
    background['Exclude'] = [1,2,3,4,5,
                            13,14,15,
                            19,20]

    archaeology = {}
    archaeology['Description'] = 'archaeology'
    archaeology['Include'] = [1,2]
    archaeology['Exclude'] = [13,14,15,
                            19,20]

    natural = {}
    natural['Description'] = 'natural'
    natural['Include'] = [13,14]
    natural['Exclude'] = [1,2,3,4,5,
                        19,20]

    modern = {}
    modern['Description'] = 'modern'
    modern['Include'] = [20]
    modern['Exclude'] = [1,2,3,4,5,
                        13,14,15]

    class_definitions = [background,archaeology,natural,modern]

    return(class_definitions)

def grid_project(polygons,grid_size = 10):
    minx = np.min(polygons.bounds['minx'])
    maxx = np.max(polygons.bounds['maxx'])
    miny = np.min(polygons.bounds['miny'])
    maxy = np.max(polygons.bounds['maxy'])

    minx_rounded = round_to_nearest_n(minx,grid_size,direction='Up')
    maxx_rounded = round_to_nearest_n(maxx,grid_size,direction='Down')
    miny_rounded = round_to_nearest_n(miny,grid_size,direction='Up')
    maxy_rounded = round_to_nearest_n(maxy,grid_size,direction='Down')


    x = np.linspace(minx_rounded,maxx_rounded, int((maxx_rounded-minx_rounded)/grid_size)+1)
    y = np.linspace(miny_rounded,maxy_rounded, int((maxy_rounded-miny_rounded)/grid_size)+1)

    return(np.meshgrid(x, y))

def round_to_nearest_n(number,n,direction='Closest'):
    if direction == 'Down':
        return math.floor(number / n) * n
    if direction == 'Up':
        return math.ceil(number / n) * n  
    if direction == 'Closest':
        return math.round(number / n) * n  
    
def plot_polygons(conn, project_id, out_dir, table = "interpretation_poly", image_name = 'polygon', image_locations = False,grid_size = 10):
    # Extract all polygons related to the projectID and put in a new table
    cur=conn.cursor()
    cur.execute("""CREATE TABLE project AS SELECT * FROM %s WHERE project_id LIKE '%s'"""%(table,project_id))
    
    fig, ax = plt.subplots(figsize=(15, 15))

    # Use Geopandas to interpret geom 
    polygons = gp.GeoDataFrame.from_postgis("SELECT * FROM project;",
        conn,
        geom_col='geom',
        coerce_float=True)
    polygons.plot(ax=ax, color='None', alpha=1)

    # Use Geopandas to interpret geom 
    polygons = gp.GeoDataFrame.from_postgis("SELECT * FROM project WHERE interp_code IN (1, 2, 3);",
        conn,
        geom_col='geom',
        coerce_float=True)
    polygons.plot(ax=ax, color='Red')

    # Use Geopandas to interpret geom 
    polygons = gp.GeoDataFrame.from_postgis("SELECT * FROM project WHERE interp_code IN (4, 5);",
        conn,
        geom_col='geom',
        coerce_float=True)
    polygons.plot(ax=ax, color='Orange')

    # Use Geopandas to interpret geom 
    polygons = gp.GeoDataFrame.from_postgis("SELECT * FROM project WHERE interp_code IN (13,14,15);",
        conn,
        geom_col='geom',
        coerce_float=True)
    polygons.plot(ax=ax, color='Green')

    # Use Geopandas to interpret geom 
    polygons = gp.GeoDataFrame.from_postgis("SELECT * FROM project WHERE interp_code IN (19,20);",
        conn,
        geom_col='geom',
        coerce_float=True)
    polygons.plot(ax=ax, color='Purple')

    # Use Geopandas to interpret geom 
    polygons = gp.GeoDataFrame.from_postgis("SELECT * FROM project WHERE interp_code IN (7,8,9);",
        conn,
        geom_col='geom',
        coerce_float=True)
    polygons.plot(ax=ax, color='Brown')

    if image_locations:
        plt.plot(image_locations[0],image_locations[1],'c.')
    plt.savefig(os.path.join(out_dir,project_id+'_'+image_name+'.png'))

    cur.execute("""DROP TABLE project""")